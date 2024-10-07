import asyncio

async def func1():
    print("Function 1 started")
    await asyncio.sleep(2)
    print("Function 1 finished")

async def func2():
    print("Function 2 started")
    await asyncio.sleep(1)
    print("Function 2 finished")

async def main():
    # Run func1 and func2 in parallel
    await asyncio.gather(func1(), func2())

# Run the asyncio event loop
asyncio.run(main())
