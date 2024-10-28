from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import os

from prompt import system_prompt_for_fetch_information_from_graph_db

class GraphSearch:
    def __init__(self, graph, client, embeddings_model, threshold=0.81):
        self.graph = graph
        self.client = client
        self.embeddings_model = embeddings_model
        self.threshold = threshold
        self.entity_types = ['Feature', 'Product', 'Price']
        self.relation_types = ['HAS_PRICE', 'HAS_FEATURE']
        self.entity_relationship_match = {'Product': ['HAS_PRICE', 'HAS_FEATURE']}

    def define_query(self, prompt, model="gpt-4-1106-preview"):
        completion = self.client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_for_fetch_information_from_graph_db.format(
                        json.dumps(self.entity_types),
                        json.dumps(self.relation_types)
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content

    def create_embedding(self, text):
        result = self.client.embeddings.create(
            model=self.embeddings_model,
            input=text
        )
        return result.data[0].embedding

    def create_query(self, query_data, threshold=0.81):
        # Print query data for debugging
        # Step 1: Creating embeddings
        embeddings_data = []
        for key, val in query_data.items():
            if key != 'Product':
                embeddings_data.append(f"${key}Embedding AS {key}Embedding")

        # Add a WITH clause if there are any embeddings; otherwise, use WITH *
        if embeddings_data:
            query = "WITH " + ",\n".join(e for e in embeddings_data)
        else:
            query = "WITH *"

        # Step 2: Matching products to each entity
        query += "\nMATCH (p:Product)\n"
        match_data = []
        for key, val in query_data.items():
            if key != 'Product':
                # Handle multiple relationships by creating separate MATCH clauses
                relationships = self.entity_relationship_match.get(key, [])
                for rel in relationships:
                    match_data.append(f"MATCH (p)-[:{rel}]->({key}Var:{key})")

        # Add relationship MATCH clauses to the query if they exist
        if match_data:
            query += "\n".join(e for e in match_data) + "\n"

        # Step 3: Filtering based on threshold (similarity could be precomputed externally)
        similarity_conditions = []
        for key, val in query_data.items():
            if key != 'Product':
                similarity_conditions.append(f"{key}Var.embedding IS NOT NULL")  # Ensure embeddings exist

        # Add a WHERE clause if there are any conditions
        if similarity_conditions:
            query += "WHERE " + " AND ".join(e for e in similarity_conditions) + "\n"

        # Step 4: Return the products
        query += "RETURN p"
        
        return query


    def query_graph(self, response_text):
        # Create embeddings for the entities
        embeddings_params = {}
        # query_data = json.loads(response_text)
        query_data = {
            "Product": "iPhone"
        }
        for key in query_data:
            embeddings_params[f"{key}Embedding"] = self.create_embedding(query_data[key])
        
        # Construct the graph query
        query = self.create_query(query_data)

        # Execute the query with the parameters
        result = self.graph.query(query, params=embeddings_params)
        return result

    @staticmethod
    def sanitize(text):
        return str(text).replace("'", "").replace('"', "").replace('{', '').replace('}', '')

    def search(self, prompt):
        # Generate the query response
        # response = self.define_query(prompt)
        # print("--- Generated Query Response ---\n", response)

        response = {
            "Product": "iPhone"
        }
        
        # Query the graph database with the generated response
        result = self.query_graph(response)
        print("--- Query Results ---\n", result)
        
        # Output the matching results
        if not result:
            print("No matching documents found.")
        else:
            print(f"Found {len(result)} matching document(s):\n")
            for r in result:
                # print(f"{r['p']['name']} ({r['p']['id']})")
                print(f"{r['p']}")