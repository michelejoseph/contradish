"""
Tests for contradish.adapters: wrap_litellm and wrap_openai_compatible.
Run with: pytest tests/test_adapters.py
No API key required.
"""
from contradish.adapters import wrap_litellm, wrap_openai_compatible


def test_wrap_litellm_missing_dep_raises_clear_importerror():
    # litellm is not installed in the test environment; the helper must raise
    # a clear, actionable ImportError rather than a bare ModuleNotFoundError.
    try:
        wrap_litellm(model="gpt-4o")
        # If litellm happens to be installed, the call returns a callable.
    except ImportError as e:
        assert "litellm" in str(e).lower()
        assert "pip install" in str(e).lower()


def test_wrap_openai_compatible_returns_callable_and_calls_client():
    captured = {}

    class FakeCompletions:
        @staticmethod
        def create(model, messages, **kw):
            captured["model"] = model
            captured["messages"] = messages
            class R:
                class Choice:
                    class Msg:
                        content = "  hello from fake  "
                    message = Msg()
                choices = [Choice()]
            return R()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    app = wrap_openai_compatible(FakeClient(), model="test-model", system="SYS")
    out = app("what is x?")
    assert out == "hello from fake"          # stripped
    assert captured["model"] == "test-model"
    # system prompt should be prepended as a system message
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["messages"][1] == {"role": "user", "content": "what is x?"}


def test_wrap_openai_compatible_no_system():
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    class R:
                        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]
                    return R()
    app = wrap_openai_compatible(FakeClient, model="m")
    assert app("q") == "ok"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
