import asyncio
import asyncpg
import os

CREDENTIALS = [
    ("postgres", "password", "lightrag"),
    ("your_username", "your_password", "your_database"),
    ("postgres", "postgres", "postgres"),
    ("postgres", "postgres", "lightrag"),
    ("lightrag", "lightrag", "lightrag"),
]

async def main():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    conn = None
    for user, password, database in CREDENTIALS:
        print(f"Trying {user}@{host}:{port}/{database}...")
        try:
            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            print("Connected!")
            break
        except Exception as e:
            print(f"Failed: {e}")
    
    if not conn:
        print("Could not connect to Postgres.")
        return

    try:
        # Load AGE extension
        try:
            await conn.execute("LOAD 'age';")
            await conn.execute("SET search_path = ag_catalog, '$user', public;")
        except Exception as e:
             print(f"Error loading AGE: {e}")

        # List graphs
        graphs = await conn.fetch("SELECT * FROM ag_catalog.ag_graph;")
        print(f"Found {len(graphs)} graphs:")
        for g in graphs:
            print(f"  - {g['name']} (ID: {g['graphid']})")
            
            # Count nodes in each graph
            graph_name = g['name']
            try:
                # Count nodes using the internal table
                count = await conn.fetchval(f"SELECT count(*) FROM \"{graph_name}\".\"_ag_label_vertex\";")
                print(f"    Nodes: {count}")
                
                if count > 0:
                    # Sample nodes
                    query = f"SELECT * FROM cypher('{graph_name}', $$ MATCH (n) RETURN n LIMIT 5 $$) as (n agtype);"
                    nodes = await conn.fetch(query)
                    print(f"    Sample nodes:")
                    for n in nodes:
                        print(f"      {n['n']}")
            except Exception as e:
                print(f"    Error querying graph {graph_name}: {e}")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
