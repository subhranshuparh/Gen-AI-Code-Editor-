from agent.llm_model import load_llm

def main():
    print("🚀 Loading TinyLlama model... (this might take a minute the first time)")
    llm = load_llm()

    # Give a simple programming prompt
    prompt = (
        "Write a Python function that returns the factorial of a number using recursion."
    )

    print("\n💬 Prompt:")
    print(prompt)
    print("\n🤖 Model Response:")

    result = llm.invoke(prompt)
    print(result)

if __name__ == "__main__":
    main()
