import os
import importlib

# Get the current directory of this package
package_directory = os.path.dirname(__file__)

# List all files in the package directory
package_files = [f for f in os.listdir(package_directory) if f.endswith(".py") and f != "__init__.py"]

# Import each module dynamically
for file_name in package_files:
    module_name = file_name[:-3]  # Remove the ".py" extension
    module_path = f"{__name__}.{module_name}"  # Construct the full module path
    importlib.import_module(module_path)

__all__ = package_files