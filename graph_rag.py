import os
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

# Load environment variables from .env file
load_dotenv()

class GraphRAG:
    def __init__(self, llms):
        self.check_and_start_neo4j()
        # Load Neo4j credentials from .env file
        neo4j_uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
        self.neo4j_username = os.getenv("NEO4J_USERNAME") or "neo4j"
        self.neo4j_password = os.getenv("NEO4J_PASSWORD") or "your_password"

        # Initialize the Neo4j graph
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )
        self.llms = llms

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
                    "-e", f"NEO4J_AUTH={self.neo4j_username}/{self.neo4j_password}",
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

            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            return True
        except Exception as e:
            st.error(f"Error while creating the graph: {str(e)}")
            return False

    def visualize_graph(
            self, 
            query = """
                MATCH (n)-[r]->(m)
                return n, r, m
                LIMIT 100
                """
        ):
        try:
            st.info('Visualizing the graph.')
            graph_data = self.graph.query(query)
            G = networkx.DiGraph()

            for record in graph_data:
                n = record['n']
                r = record['r']
                m = record['m']
                G.add_node(n['id'])
                G.add_node(m['id'])
                G.add_edge(n['id'], m['id'])

            net = Network(notebook=True)
            net.from_nx(G)
            net.show('graph.html')
            HtmlFile = open('graph.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.v1.html(source_code, height=600, scrolling=True)
            return graph_data
        except Exception as e:
            st.error(f"Error while visualizing the graph: {str(e)}")
            return None
