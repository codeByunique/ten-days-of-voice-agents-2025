from src.agent import save_order_local
import asyncio
asyncio.run(save_order_local("cappuccino","large","oat",["chocolate"],"Ataul"))
print("done")
