import os
import json
import subprocess
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
import networkx
from networkx import Graph
from pyvis.network import Network
from streamlit import components
import traceback
# from prompt import prompt_to_extract_entities_and_relationships



def node_to_string(node):
    return f"Node(id='{node.id}', type='{node.type}', properties={node.properties})"

def relationship_to_string(relationship):
    source_str = node_to_string(relationship.source)
    target_str = node_to_string(relationship.target)
    return f"Relationship(source={source_str}, target={target_str}, type='{relationship.type}', properties={relationship.properties})"

def graph_document_to_string(graph_document):
    # Convert nodes to strings
    nodes_str = ', '.join([node_to_string(node) for node in graph_document.nodes])
    # Convert relationships to strings
    relationships_str = ', '.join([relationship_to_string(rel) for rel in graph_document.relationships])
    # Format the source document
    source_str = f"Document(metadata={graph_document.source.metadata}, page_content='{graph_document.source.page_content}')"
    # Combine all parts
    return f"GraphDocument(nodes=[{nodes_str}], relationships=[{relationships_str}], source={source_str})"

# Convert a list of GraphDocument objects to a string representation
def graph_documents_to_string(graph_documents):
    result = []
    for doc in graph_documents:
        doc_str = graph_document_to_string(doc)
        print('---', doc_str)
        result.append(doc_str)
    return ', '.join(result)


# Load environment variables from .env file
load_dotenv()

class GraphRAG:
    def __init__(self, llms):
        # Load Neo4j credentials from .env file
        neo4j_uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
        self.neo4j_username = os.getenv("NEO4J_USERNAME") or "neo4j"
        self.neo4j_password = os.getenv("NEO4J_PASSWORD") or "your_password"

        print('------------------', neo4j_uri, self.neo4j_username, self.neo4j_password)

        self.check_and_start_neo4j()
        # Initialize the Neo4j graph
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )

        self.llms = llms

    def extract_entities_and_relationships(self, graph_documents):
        entities = []
        relationships = []

        for graph_doc in graph_documents:
            # Extract nodes dynamically based on available keys
            for node in graph_doc.nodes:
                node_data = {'id': node.id, 'type': node.type}
                # Add all properties from the node if they exist
                if node.properties:
                    node_data.update(node.properties)
                entities.append(node_data)

            # Extract relationships dynamically
            for relationship in graph_doc.relationships:
                rel_data = {
                    'source': relationship.source.id,
                    'target': relationship.target.id,
                    'type': relationship.type
                }
                # Add all properties from the relationship if they exist
                if relationship.properties:
                    rel_data.update(relationship.properties)
                relationships.append(rel_data)

        return entities, relationships

    def check_and_start_neo4j(self):
        try:
            # Ensure Docker is installed
            docker_check = subprocess.run(
                ["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if docker_check.returncode != 0:
                st.error("Docker is not installed or not available.")
                return

            # Check if Neo4j container is running
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", "neo4j"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if "true" not in result.stdout:
                # Stop and remove any existing Neo4j container named "neo4j"
                subprocess.run(["docker", "rm", "-f", "neo4j"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.info("Neo4j server is not running. Starting Neo4j using Docker...")
                # Start Neo4j if it's not running
                start_result = subprocess.run([
                    "docker", "run", "-d", "--name", "neo4j",
                    "-p", "7687:7687", "-p", "7474:7474",
                    "-e", f"NEO4J_AUTH=neo4j/{self.neo4j_password}",
                    "-e", 'NEO4J_PLUGINS=["apoc"]',
                    "neo4j"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if start_result.returncode == 0:
                    st.success("Neo4j server started successfully.")
                    self.wait_for_neo4j_to_be_ready()
                else:
                    st.error(f"Failed to start Neo4j. Error: {start_result.stderr}")
            else:
                st.success("Neo4j server is already running.")
                self.wait_for_neo4j_to_be_ready()
        except Exception as e:
            st.error(f"Error while checking or starting Neo4j: {e}")

    def wait_for_neo4j_to_be_ready(self, max_retries=30, delay=2):
        """Wait for Neo4j to be fully ready."""
        st.info("Waiting for Neo4j to be ready...")
        for attempt in range(max_retries):
            try:
                # Try to connect to Neo4j and run a simple query
                result = subprocess.run(
                    ["docker", "exec", "neo4j", "cypher-shell", "-u", "neo4j", "-p", os.getenv("NEO4J_PASSWORD"), "RETURN 1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode == 0:
                    local_ui = os.getenv("NEO4J_LOCAL_UI") or "http://localhost:7474"
                    st.success("Neo4j is ready. Check the Neo4j browser at [here]({}).".format(local_ui))
                    return
                else:
                    st.warning(f"Neo4j is not ready yet. Retrying in {delay} seconds...")
            except Exception as e:
                st.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(delay)

        st.error("Neo4j did not become ready in time. Please check the Docker container logs.")

    def create_graph(self, docs):
        try:
            # st.info('Converting documents to graph format.')
            graph_documents = LLMGraphTransformer(self.llms).convert_to_graph_documents(docs)
            st.success('Documents converted to graph format successfully.')

            print('graph_documents', graph_documents)

            # graph_documents_str = graph_documents_to_string(graph_documents)

            # print("&&&graph_documents_str", graph_documents_str)

            # prompt_to_extract_entities_and_relationships = """
            #     From the array:
            #     {}

            #     Extract three variables as follows:

            #     entity_types = {{
            #         "product": "Item detailed type, such as 'iPhone 11', 'Moto G Stylus 2023'.",
            #         "brand": "The brand associated with the product, for example 'Apple', 'Cricket Wireless'.",
            #         "specification": "Product specifications like '64GB', '4GB RAM', '8MP FF Camera'.",
            #         "color": "The color of the product, for example 'Black', 'Blue'.",
            #         "product_type": "General category or type of the product, such as 'Prepaid Smartphone'.",
            #         "organization": "Company or organization related to the product, like 'AT&T' or 'Apple'."
            #     }}

            #     relation_types = {{
            #         "produces": "Organization produces the product.",
            #         "features": "Product has certain features or specifications.",
            #         "color": "Product is of a certain color.",
            #         "sells": "Organization sells the product.",
            #         "brand": "The brand associated with the product.",
            #         "has_specification": "Product has a particular specification.",
            #         "is_a": "Product belongs to a general product type."
            #     }}

            #     entity_relationship_match = {{
            #         "organization": "produces",
            #         "product": "features",
            #         "color": "color",
            #         "brand": "brand",
            #         "specification": "has_specification",
            #         "product_type": "is_a"
            #     }} 

            #     Output Format: Return only three variables in an Python array. No wrapped in markdown. No further explanation.
            #     """.format(str(graph_documents_str))




            # entities_and_relationships = self.llms.invoke(
            #     prompt_to_extract_entities_and_relationships,
            # )

            # print('===entities_and_relationships', entities_and_relationships.content)

            # parsed_globals = {}
            # exec(entities_and_relationships.content, parsed_globals)

            # entity_types = parsed_globals['entity_types']
            # relation_types = parsed_globals['relation_types']
            # entity_relationship_match = parsed_globals['entity_relationship_match']


            # print('--entity_types', entity_types)
            # print('--relation_types', relation_types)
            # print('--entity_relationship_match', entity_relationship_match)

            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
        except Exception as e:
            st.error(f"Error while creating the graph: {str(e)}")
            st.write(traceback.format_exc())
        
    def query_graph(self, query):
        try:
            return self.graph.query(query)
        except Exception as e:
            st.error(f"Error while querying the graph: {str(e)}")

    
    def visualize_graph(
            self, 
            data,
        ):
        try:
            G = networkx.DiGraph()
            for record in data:
                # Check if 'n', 'r', and 'm' exist in the record
                if 'n' in record:
                    n = record['n']
                else:
                    continue

                if 'r' in record:
                    r = record['r']

                if 'm' in record:
                    m = record['m']
                else:
                    continue

                # Check if 'id' exists in both 'n' and 'm' before adding nodes
                if 'id' in n:
                    G.add_node(n['id'])

                if 'id' in m:
                    G.add_node(m['id'])

                # Add edge only if both 'id' exist in 'n' and 'm'
                if 'id' in n and 'id' in m:
                    G.add_edge(n['id'], m['id'])

            net = Network(notebook=True)
            net.from_nx(G)
            net.show('graph.html')
            HtmlFile = open('graph.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.v1.html(source_code, height=600, scrolling=True)

        except Exception as e:
            st.error(f"Error while visualizing the graph: {str(e)}")
            return None
