"""第3回: Pydantic AI で作る学習支援エージェント。

使い方:
    uv run python -m app.agent "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"
    uv run python -m app.agent --trace "LoRAとQLoRAの違いは？"   # ツール呼び出しを表示
    uv run python -m app.agent --mcp --trace "LoRAとは？"        # 第5回: ツールを MCP サーバーから受け取る
    uv run python -m app.agent --mcp --list-tools               # 第5回: モデルに見えるツール一覧（APIキー不要）

エージェント = モデル + ループ + ツール。
モデルは tools.py の関数を「ツール」として呼び出し、その結果を見て回答を作る。
第5回では、同じツールを MCP サーバー（app/mcp_server.py）経由で受け取る。
"""

import argparse
import inspect
import json
from pathlib import Path

from pydantic_ai import Agent, DeferredToolRequests, ToolApproved, ToolDenied
from pydantic_ai.exceptions import ModelHTTPError
from fastmcp import Client
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.google import GoogleProvider

from app import tools
from app.config import MODEL, api_error_message, get_api_key

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""

# モデルは実行時に渡す（テストではAPIを呼ばない TestModel に差し替えるため）
agent = Agent(instructions=INSTRUCTIONS, output_type=[str, DeferredToolRequests])
agent.tool_plain(tools.search_glossary)
agent.tool_plain(tools.get_lecture_summary)

# TODO(第3回 演習): tools.py に make_quiz を作り、ここで登録しよう
agent.tool_plain(tools.make_quiz)

# ログ保存するやつ（第３回宿題）
#agent.tool_plain(tools.save_learning_log, requires_approval=True)
agent.tool(tools.save_learning_log, requires_approval=True)


# 第5回: 使う MCP サーバーの設定ファイル。Claude Desktop や Cursor と同じ "mcpServers" 形式
MCP_CONFIG = Path(__file__).resolve().parent / "mcp_servers.json"


def load_mcp_servers(config_path: Path = MCP_CONFIG) -> list[MCPToolset]:
    """設定ファイル（mcpServers 形式）を読み、サーバーごとに接続（MCPToolset）を作る。

    - "command" があれば stdio：サーバーをサブプロセスとして起動する
    - "url" があれば Streamable HTTP：起動済みのサーバーに接続する
    ツール名には「サーバー名_」を付ける（例: glossary_search_glossary）。複数のサーバーに
    同じ名前のツールがあっても区別できるようにするため。

    Pydantic AI には同じことをする load_mcp_toolsets() もある。ここで自分で組み立てているのは、
    古い SDK で作られたサーバー（fetch など）につなぐとき、接続方式の自動判定（"auto"）が
    警告を大量に出すため。"legacy" に固定すると新旧どちらのサーバーにも静かにつながる。
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))["mcpServers"]
    toolsets = []
    for name, spec in config.items():
        if "url" in spec:
            transport = spec["url"]
        else:
            transport = StdioTransport(spec["command"], spec.get("args", []), env=spec.get("env"))
        toolsets.append(MCPToolset(Client(transport, mode="legacy")).prefixed(name))
    return toolsets


def build_mcp_agent(config_path: Path = MCP_CONFIG) -> Agent:
    """第5回: 関数を直接登録する代わりに、MCP サーバーが公開するツールを使うエージェント。INSTRUCTIONS は同じ。"""
    return Agent(
        instructions=INSTRUCTIONS,
        output_type=[str, DeferredToolRequests],
        toolsets=load_mcp_servers(config_path),
    )


def build_model() -> GoogleModel:
    return GoogleModel(MODEL, provider=GoogleProvider(api_key=get_api_key()))


def print_tools(target: Agent) -> None:
    """モデルに見えているツールの一覧（名前・説明文・引数）を表示する。LLM は呼ばない。

    TestModel（テストで使う偽モデル）にツール一覧を渡し、そこから読み出す。
    MCP サーバーのツールも、第3回の関数ツールも、モデルから見れば同じ形をしている。
    """
    model = TestModel(call_tools=[])
    target.run_sync("ツール一覧", model=model)
    tools_seen = model.last_model_request_parameters.function_tools
    print(f"モデルに見えているツール: {len(tools_seen)} 個")
    for tool in tools_seen:
        print(f"\n[{tool.name}]")
        print("  " + inspect.cleandoc(tool.description or "").replace("\n", "\n  "))
        params = tool.parameters_json_schema.get("properties", {})
        print(f"  引数: {json.dumps(params, ensure_ascii=False)}")


def print_trace(result) -> None:
    """どのツールが、どの引数で呼ばれ、何を返したかを表示する。"""
    print("\n--- trace ---")
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"[tool call]   {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                print(f"[tool return] {part.tool_name} -> {str(part.content)[:120]}…")
    print(f"[usage] {result.usage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="学習支援エージェントに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--trace", action="store_true", help="ツール呼び出しを表示する")
    parser.add_argument("--mcp", action="store_true", help="第5回: ツールを MCP サーバー（app/mcp_servers.json）から受け取る")
    parser.add_argument("--list-tools", action="store_true", help="モデルに見えるツール一覧を表示して終了する（APIキー不要）")
    args = parser.parse_args()

    target = build_mcp_agent() if args.mcp else agent
    if args.list_tools:
        print_tools(target)
        return
    if not args.question:
        #parser.error("質問を指定してください")
        chat(target=target)
        return

    print(f"[model] {MODEL}")
    if args.mcp:
        print("[tools] MCP サーバー（app/mcp_servers.json）")
    try:
        result = target.run_sync(args.question, model=build_model())
    except ModelHTTPError as e:
        print(api_error_message(e.status_code, str(e.body)[:200]))
        raise SystemExit(1)
    print(result.output)
    if args.trace:
        print_trace(result)

# ログ保存の確認をするための対話モード
def is_hint_request(text: str) -> bool:
    normalized = text.casefold().strip()
    return (
        normalized in {"ヒント", "hint", "help"}
        or normalized.startswith(("ヒントを", "ヒントください", "ヒントをください", "ヒント出して"))
    )


def chat(target: Agent | None = None) -> None:
    print("=== 学習支援エージェントと対話を開始します ('exit' または 'quit' で終了) ===")
    print("確認問題は '/quiz 2' のように指定してください。")

    target = target or agent
    message_history = []  # これまでの会話の履歴（セッション）を保持するリスト
    quiz = None           # 現在のクイズデータを一時保持

    while True:
        try:
            question = input("\n[質問] ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します。")
            break

        # 終了コマンドの判定
        if question.lower() in {"exit", "quit"}:
            print("お疲れ様でした！")
            break
        if not question:
            continue

        # クイズを作る専用コマンド '/quiz N' の処理
        if question.startswith("/quiz"):
            try:
                lecture = int(question.split(maxsplit=1)[1])
                quiz_data = json.loads(tools.make_quiz(lecture))
                
                # エラー処理
                if quiz_data["answer"] in {"", "Error"}:
                    print(quiz_data["question"])
                    continue
                
                # 正常処理
                quiz = quiz_data
                print(f"\n第{lecture}回 確認問題")
                print(quiz_data["question"])
                if quiz_data.get("hint"):
                    print(f"ヒント: {quiz_data['hint']}")
                print("回答を入力してください。")
                continue

            except (IndexError, ValueError, json.JSONDecodeError):
                print("使い方: /quiz 2  （第2回の保存確認問題を出題）")
                continue

        # クイズが出題された状態でユーザーが「答え」を入力したときの処理
        if quiz is not None:
            if is_hint_request(question):
                print(f"ヒント: {quiz.get('hint', '用語集のカテゴリーや関連用語を手掛かりに考えてみましょう。')}")
                continue

            expected = quiz["answer"]

            # 表記揺れ（全角半角やスペース）を防ぐために文字列を正規化して比較
            normalized_answer = question.casefold().replace(" ", "").replace(" ", "")
            normalized_expected = expected.casefold().replace(" ", "").replace(" ", "")
            is_correct = bool(normalized_expected) and normalized_answer == normalized_expected
            
            print("正解です！" if is_correct else "残念、不正解です。")
            print(f"正解: {expected}")

            # 学習ログへの保存確認
            save = input("この問題と回答を学習ログに保存しますか？ [y/N] ").strip().lower()
            if save in {"y", "yes", "はい"}:
                lecture = f"第{quiz['lecture']}回"
                quiz_question = quiz["question"]
                answer_text = f"ユーザー回答: {question} / 正解: {expected}"
                
                # 保存を指示するプロンプト
                prompt = (
                    f"第{quiz['lecture']}回の問題『{quiz_question}』と回答『{answer_text}』を"
                    "save_learning_logで保存してください。"
                )
                try:
                    # 会話を引き継いで、エージェントを実行
                    result = target.run_sync(
                        prompt,
                        message_history=message_history,
                        model=build_model(),
                    )
                    message_history = result.all_messages() # 履歴を更新

                    # requires_approval=True により、承認待ちになった呼び出しだけを再開する。
                    if isinstance(result.output, DeferredToolRequests):
                        save_calls = [
                            call for call in result.output.approvals
                            if call.tool_name == "save_learning_log"
                        ]
                        if save_calls:
                            approvals = {
                                call.tool_call_id: ToolApproved()
                                for call in save_calls
                            }
                            result = target.run_sync(
                                message_history=message_history,
                                deferred_tool_results=result.output.build_results(approvals=approvals),
                                model=build_model(),
                            )
                            message_history = result.all_messages()

                            saved = any(
                                isinstance(part, ToolReturnPart)
                                and part.tool_name == "save_learning_log"
                                for message in result.all_messages()
                                for part in message.parts
                            )
                            print("学習ログを保存しました。" if saved else "学習ログの保存処理が完了しませんでした。")
                        else:
                            print("save_learning_logの承認待ち呼び出しが見つかりませんでした。")
                    else:
                        print("学習ログ保存ツールが呼び出されませんでした。")

                    print(f"\n{result.output}")
                except Exception as e:
                    print(f"保存に失敗しました: {e}")
            else:
                print("学習ログへの保存をキャンセルしました。")

            quiz = None  # クイズの状態をリセット
            continue

        # 通常のフリー会話・質問の処理
        try:
            result = target.run_sync(
                question,
                message_history=message_history,
                model=build_model(),
            )
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            continue

        print(f"\n学習支援エージェント:\n{result.output}")
        message_history = result.all_messages()  # 会話履歴を更新して次のループへ


if __name__ == "__main__":
    main()
