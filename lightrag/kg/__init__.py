print ("init package vars here. ......")
# from .neo4j import GraphStorage as Neo4JStorage


# import sys
# import importlib
# # Specify the path to the directory containing the module
# # Add the directory to the system path
# module_dir = '/Users/kenwiltshire/documents/dev/LightRag/lightrag/kg'
# sys.path.append(module_dir)
# # Specify the module name
# module_name = 'neo4j'
# # Import the module
# spec = importlib.util.spec_from_file_location(module_name, f'{module_dir}/{module_name}.py')

# Neo4JStorage = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(Neo4JStorage)



# Relative imports are still possible by adding a leading period to the module name when using the from ... import form:

# # Import names from pkg.string
# from .string import name1, name2
# # Import pkg.string
# from . import string

