from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from fastapi_mcp import FastApiMCP

# --- 1. Define your FastAPI App and Endpoints ---
app = FastAPI(
    title="My Items API",
    description="A simple API to manage items."
)

class Item(BaseModel):
    id: int
    name: str
    price: float

items_db = {
    1: Item(id=1, name="Hammer", price=9.99),
    2: Item(id=2, name="Screwdriver", price=5.50),
}

@app.get("/items/", response_model=List[Item], operation_id="list_items")
def list_items():
    """List all available items."""
    return list(items_db.values())

@app.get("/items/{item_id}", response_model=Item, operation_id="get_item")
def get_item(item_id: int):
    """Get a specific item by its ID."""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

# --- 2. Create and Mount the MCP Server ---
mcp = FastApiMCP(app)
mcp.mount_http() # Mounts at the default /mcp path

# --- 3. Add a Runner (for local execution) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)