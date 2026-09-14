"""ui_output_check.py —— 事件回调「返回值数量 vs 输出组件数量」静态检查。

背景：Gradio 5 里回调返回的元组长度必须与绑定的 outputs 数量一致，
少了直接抛 "A function didn't return enough output values"（训练页
on_poll 的 3 vs 4 事故就是这一类，Timer 每 2.5s 抛一次）。这类错误
build_check（只验证 Tab 画得出来）和 stage2_ui_probe（不触发 Timer）
都拦不住，所以在这里做**源码级**检查：

  1. 扫出全部 `.click/.change/.tick/.select/...` 事件绑定；
  2. 回调是本文件内定义的函数时，数它每条 return 路径的元素个数
     （含 `*(...)` 解包的按「未知」处理）；
  3. 任何**已知**分支的个数与 outputs 数不一致就报错；
     全部分支都是未知（动态构造）的绑定跳过并提示人工复核；
  4. 顺带核对 inputs 数量不超过回调的位置参数个数（多了会在
     运行时 TypeError）；
  5. 各 Tab 返回的 `page_load: (fn, [outputs])` 也按同样规则核对。

只做静态分析、不启动 Gradio，秒级完成。

跑法：  .venv\\Scripts\\python.exe tools\\ui_output_check.py
退出码：0 = 干净；1 = 有 ERROR。
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Gradio Blocks 组件的事件方法（绑定回调用的）
EVENT_METHODS = {
    "click", "change", "tick", "select", "submit", "input", "upload",
    "load", "key_up", "clear", "delete", "edit", "like", "share",
    "focus", "blur", "double_click", "play", "pause", "stop", "end",
    "start_recording", "pause_recording", "stop_recording", "reveal",
    "expand", "collapse", "apply",
}

TARGETS = list((PROJECT_ROOT / "webui_app" / "tabs").glob("*.py")) + [
    PROJECT_ROOT / "webui_app" / "app.py",
]


def _iter_returns(nodes):
    """递归产出 Return 节点，但**不进入**嵌套函数/lambda —— 回调体内定义的
    内层函数（on_extract 里的 fn(progress, should_stop) 是常事）有自己的
    return，算进外层会误报。"""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Return):
            yield node
        yield from _iter_returns(ast.iter_child_nodes(node))


def return_counts(fn: ast.FunctionDef | ast.AsyncFunctionDef):
    """收集一个函数所有 return 路径的元素个数。

    返回 (known, has_unknown)：known 是确定个数的集合；
    has_unknown 表示存在动态解包/委托调用等无法静态确定的分支。
    """
    known, has_unknown = set(), False
    for node in _iter_returns(ast.iter_child_nodes(fn)):
        v = node.value
        if v is None:                      # 裸 return：视作 1 个 None
            known.add(1)
        elif isinstance(v, ast.Tuple):
            if any(isinstance(e, ast.Starred) for e in v.elts):
                has_unknown = True
            else:
                known.add(len(v.elts))
        elif isinstance(v, ast.Call):
            has_unknown = True             # 委托给别的函数，长度跟过去算
        elif isinstance(v, ast.IfExp):
            has_unknown = True
        else:
            known.add(1)
    return known, has_unknown


def param_capacity(fn) -> int | None:
    """回调能吃下的位置参数上限；*args 视为不限（None）。"""
    total = 0
    for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs:
        total += 1
    if fn.args.vararg is not None:
        return None
    return total


def outputs_len(node: ast.expr) -> int | None:
    """outputs= 的长度；非字面列表返回 None（无法静态确定）。"""
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    return None


def inputs_len(node: ast.expr) -> int | None:
    return outputs_len(node)


class Checker(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.notes: list[str] = []
        self.functions: dict[str, ast.FunctionDef] = {}
        self.lambdas: dict[int, ast.Lambda] = {}

    def visit_Module(self, node):
        for sub in ast.walk(node):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[sub.name] = sub
        self.generic_visit(node)

    # ---- 事件绑定：xxx.click(fn, inputs=[...], outputs=[...]) ----
    def visit_Call(self, node):
        func = node.func
        if (isinstance(func, ast.Attribute)
                and func.attr in EVENT_METHODS
                and isinstance(func.value, ast.Attribute | ast.Name)):
            self._check_binding(node)
        self.generic_visit(node)

    def _resolve_fn(self, fn_arg: ast.expr):
        """把回调表达式解析成 (FunctionDef, 描述)；解析不了返回 None。"""
        if isinstance(fn_arg, ast.Name):
            f = self.functions.get(fn_arg.id)
            return (f, fn_arg.id) if f else None
        if isinstance(fn_arg, ast.Lambda):
            return fn_arg, "<lambda>"
        # lambda: on_poll(False) —— 跟进被调函数
        if (isinstance(fn_arg, ast.Lambda)
                and isinstance(fn_arg.body, ast.Call)):
            pass
        return None

    def _check_binding(self, call: ast.Call):
        where = f"{self.path.name}:{call.lineno}"
        fn_arg = call.args[0] if call.args else None
        kw = {k.arg: k.value for k in call.keywords}
        outs_node = kw.get("outputs") or (call.args[2] if len(call.args) > 2
                                          else None)
        ins_node = kw.get("inputs") or (call.args[1] if len(call.args) > 1
                                        else None)
        if fn_arg is None or outs_node is None:
            return
        n_out = outputs_len(outs_node)
        n_in = inputs_len(ins_node) if ins_node is not None else 0
        method = call.func.attr

        # lambda 直接数 body；lambda: f(x) 跟进 f
        if isinstance(fn_arg, ast.Lambda):
            body = fn_arg.body
            if isinstance(body, ast.Tuple):
                if any(isinstance(e, ast.Starred) for e in body.elts):
                    self.notes.append(
                        f"{where} [{method}] lambda 返回含 *解包，请人工复核"
                        f"（outputs={n_out}）")
                    return
                if n_out is not None and len(body.elts) != n_out:
                    self.errors.append(
                        f"{where} [{method}] lambda 返回 {len(body.elts)} 个值，"
                        f"但 outputs 有 {n_out} 个")
                return
            if isinstance(body, ast.Call) and isinstance(body.func, ast.Name):
                target = self.functions.get(body.func.id)
                if target is not None:
                    self._compare(where, method, target, body.func.id,
                                  n_out, n_in)
                    return
            if n_out is not None and n_out != 1:
                self.errors.append(
                    f"{where} [{method}] lambda 返回单值，但 outputs 有 "
                    f"{n_out} 个")
            return

        if isinstance(fn_arg, ast.Name) and fn_arg.id in self.functions:
            self._compare(where, method, self.functions[fn_arg.id],
                          fn_arg.id, n_out, n_in)

    def _compare(self, where, method, fn, name, n_out, n_in):
        if n_out is None:
            self.notes.append(
                f"{where} [{method}] {name} 的 outputs 非字面列表，跳过")
            return
        known, has_unknown = return_counts(fn)
        bad = {k for k in known if k != n_out}
        if bad:
            self.errors.append(
                f"{where} [{method}] {name} 有的分支只返回 "
                f"{sorted(bad)} 个值（outputs 需要 {n_out} 个）——"
                "Gradio 会在运行时抛 didn't return enough output values")
        elif not known and has_unknown:
            self.notes.append(
                f"{where} [{method}] {name} 返回值动态构造，请人工复核"
                f"（outputs={n_out}）")
        cap = param_capacity(fn)
        if cap is not None and n_in is not None and n_in > cap:
            self.errors.append(
                f"{where} [{method}] {name} 只有 {cap} 个位置参数，"
                f"却绑了 {n_in} 个 inputs（运行时 TypeError）")


def check_page_load(path: Path, tree, errors: list[str], notes: list[str]):
    """各 Tab render() 返回的 page_load: (fn, [outputs]) 元组核对。"""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
            continue
        for kv in node.value.keys:
            if not (isinstance(kv, ast.Constant) and kv.value == "page_load"):
                continue
            v = node.value.values[node.value.keys.index(kv)]
            if not (isinstance(v, ast.Tuple) and len(v.elts) == 2):
                continue
            fn_ref, outs_ref = v.elts
            if not isinstance(outs_ref, (ast.List, ast.Tuple)):
                continue
            n_out = len(outs_ref.elts)
            where = f"{path.name}:{node.lineno}"
            if isinstance(fn_ref, ast.Name):
                # 找函数定义数返回值
                for sub in ast.walk(tree):
                    if (isinstance(sub, ast.FunctionDef)
                            and sub.name == fn_ref.id):
                        known, has_unknown = return_counts(sub)
                        bad = {k for k in known if k != n_out}
                        if bad:
                            errors.append(
                                f"{where} [page_load] {fn_ref.id} 有的分支返回"
                                f" {sorted(bad)} 个值（输出组件 {n_out} 个）")
                        elif not known and has_unknown:
                            notes.append(
                                f"{where} [page_load] {fn_ref.id} 返回值动态"
                                f"构造，请人工复核（outputs={n_out}）")
                        break
            elif isinstance(fn_ref, ast.Lambda):
                body = fn_ref.body
                if isinstance(body, ast.Tuple):
                    if any(isinstance(e, ast.Starred) for e in body.elts):
                        notes.append(f"{where} [page_load] lambda 含 *解包，"
                                     "请人工复核")
                    elif len(body.elts) != n_out:
                        errors.append(
                            f"{where} [page_load] lambda 返回 "
                            f"{len(body.elts)} 个值（输出组件 {n_out} 个）")


def main() -> int:
    errors: list[str] = []
    notes: list[str] = []
    n_bindings = 0
    for path in TARGETS:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        chk = Checker(path)
        chk.visit(tree)
        n_bindings += len([n for n in ast.walk(tree)
                           if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute)
                           and n.func.attr in EVENT_METHODS])
        errors += chk.errors
        notes += chk.notes
        check_page_load(path, tree, errors, notes)

    print(f"检查 {len(TARGETS)} 个文件 · {n_bindings} 个事件绑定")
    for n in notes:
        print(f"  ℹ️ {n}")
    if errors:
        print(f"\n✖ {len(errors)} 个问题：")
        for e in errors:
            print(f"  ERROR {e}")
        return 1
    print("✅ 全部事件绑定的返回值数量与输出组件匹配")
    return 0


if __name__ == "__main__":
    sys.exit(main())
