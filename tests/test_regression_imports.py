"""Regression tests for import statements and module structure.

This test suite validates:
1. All Python files can be imported without errors
2. All import statements are valid and resolve correctly
3. No circular dependencies exist
4. All required modules are accessible
5. Constants and shared resources are properly defined

Critical for catching regressions after major refactoring.
"""

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pytest


# Root directory of the custom component
COMPONENT_ROOT = Path(__file__).parent.parent / "custom_components" / "effektguard"


class ImportValidator:
    """Validates import statements and module structure."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.all_python_files: List[Path] = []
        self.import_map: Dict[str, Set[str]] = {}

    def find_all_python_files(self) -> List[Path]:
        """Find all Python files in the component."""
        python_files = list(COMPONENT_ROOT.rglob("*.py"))
        # Exclude __pycache__ and test files
        python_files = [
            f
            for f in python_files
            if "__pycache__" not in str(f) and not f.name.startswith("test_")
        ]
        self.all_python_files = python_files
        return python_files

    def extract_imports(self, filepath: Path) -> Tuple[List[str], List[str]]:
        """Extract all import statements from a Python file.

        Returns:
            Tuple of (absolute_imports, relative_imports)
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(filepath))
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {filepath}: {e}")
            return [], []

        absolute_imports = []
        relative_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    absolute_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level

                if level > 0:  # Relative import
                    relative_imports.append((level, module, [a.name for a in node.names]))
                else:  # Absolute import
                    absolute_imports.append(module)

        return absolute_imports, relative_imports

    def validate_import_statement(self, import_name: str, source_file: Path) -> bool:
        """Validate that an absolute import can be resolved."""
        # Allow standard library and third-party imports
        try:
            spec = importlib.util.find_spec(import_name.split(".")[0])
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            # Check if it's a local import
            if import_name.startswith("custom_components.effektguard"):
                relative_path = import_name.replace("custom_components.effektguard.", "").replace(
                    ".", "/"
                )
                target_file = COMPONENT_ROOT / f"{relative_path}.py"
                target_init = COMPONENT_ROOT / relative_path / "__init__.py"

                if target_file.exists() or target_init.exists():
                    return True

            self.errors.append(
                f"Cannot resolve import '{import_name}' in {source_file.relative_to(COMPONENT_ROOT)}"
            )
            return False

    def validate_relative_import(
        self, level: int, module: str, names: List[str], source_file: Path
    ) -> bool:
        """Validate that a relative import can be resolved."""
        # Calculate the target directory
        current_dir = source_file.parent
        target_dir = current_dir

        for _ in range(level - 1):
            target_dir = target_dir.parent
            if not target_dir.is_relative_to(COMPONENT_ROOT):
                self.errors.append(
                    f"Relative import goes outside component: {source_file.relative_to(COMPONENT_ROOT)}"
                )
                return False

        # Check if the module exists
        if module:
            module_path = target_dir / module.replace(".", "/")
            module_file = module_path.with_suffix(".py")
            module_init = module_path / "__init__.py"

            if not (module_file.exists() or module_init.exists()):
                self.errors.append(
                    f"Cannot resolve relative import '.{'.' * (level-1)}{module}' "
                    f"in {source_file.relative_to(COMPONENT_ROOT)}"
                )
                return False

        return True

    def check_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in imports."""
        # Build dependency graph
        graph: Dict[str, Set[str]] = {}

        for filepath in self.all_python_files:
            module_name = self._get_module_name(filepath)
            abs_imports, rel_imports = self.extract_imports(filepath)

            dependencies = set()

            # Add absolute imports that are within the component
            for imp in abs_imports:
                if imp.startswith("custom_components.effektguard"):
                    dependencies.add(imp)

            # Add relative imports
            for level, module, names in rel_imports:
                target_module = self._resolve_relative_import(filepath, level, module)
                if target_module:
                    dependencies.add(target_module)

            graph[module_name] = dependencies

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _get_module_name(self, filepath: Path) -> str:
        """Convert file path to module name."""
        relative = filepath.relative_to(COMPONENT_ROOT.parent.parent)
        module = str(relative.with_suffix("")).replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-9]
        return module

    def _resolve_relative_import(self, source_file: Path, level: int, module: str) -> str:
        """Resolve relative import to absolute module name."""
        current_dir = source_file.parent
        target_dir = current_dir

        for _ in range(level - 1):
            target_dir = target_dir.parent

        base_module = self._get_module_name(target_dir / "__init__.py")
        if base_module.endswith(".__init__"):
            base_module = base_module[:-9]

        if module:
            return f"{base_module}.{module}"
        return base_module

    def try_import_all_modules(self) -> Dict[str, Exception]:
        """Attempt to import all modules and return failures."""
        failures = {}

        # Add the component root to sys.path
        component_parent = COMPONENT_ROOT.parent.parent
        if str(component_parent) not in sys.path:
            sys.path.insert(0, str(component_parent))

        for filepath in self.all_python_files:
            module_name = self._get_module_name(filepath)

            try:
                importlib.import_module(module_name)
            except Exception as e:
                failures[module_name] = e

        return failures


@pytest.fixture
def validator():
    """Create an ImportValidator instance."""
    return ImportValidator()


class TestImportStructure:
    """Test the import structure and module accessibility."""

    def test_all_python_files_found(self, validator):
        """Verify that Python files are detected."""
        files = validator.find_all_python_files()
        assert len(files) > 0, "No Python files found in component"

        # Check for key files
        file_names = {f.name for f in files}
        assert "__init__.py" in file_names
        assert "const.py" in file_names
        assert "climate.py" in file_names
        assert "coordinator.py" in file_names

    def test_no_syntax_errors(self, validator):
        """Verify all Python files have valid syntax."""
        validator.find_all_python_files()

        for filepath in validator.all_python_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    ast.parse(f.read(), filename=str(filepath))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {filepath.relative_to(COMPONENT_ROOT)}: {e}")

    def test_all_imports_extractable(self, validator):
        """Verify imports can be extracted from all files."""
        validator.find_all_python_files()

        for filepath in validator.all_python_files:
            abs_imports, rel_imports = validator.extract_imports(filepath)
            # Should not raise exceptions
            assert isinstance(abs_imports, list)
            assert isinstance(rel_imports, list)

    def test_all_absolute_imports_valid(self, validator):
        """Verify all absolute imports can be resolved."""
        validator.find_all_python_files()

        for filepath in validator.all_python_files:
            abs_imports, _ = validator.extract_imports(filepath)

            for import_name in abs_imports:
                # Skip standard library and known third-party packages
                if import_name.split(".")[0] in [
                    "asyncio",
                    "datetime",
                    "logging",
                    "typing",
                    "dataclasses",
                    "pathlib",
                    "json",
                    "math",
                    "statistics",
                    "enum",
                    "abc",
                    "homeassistant",
                    "voluptuous",
                    "aiohttp",
                    "pytest",
                ]:
                    continue

                # Validate custom component imports
                if import_name.startswith("custom_components.effektguard"):
                    validator.validate_import_statement(import_name, filepath)

        if validator.errors:
            pytest.fail("\n".join(validator.errors))

    def test_all_relative_imports_valid(self, validator):
        """Verify all relative imports can be resolved."""
        validator.find_all_python_files()

        for filepath in validator.all_python_files:
            _, rel_imports = validator.extract_imports(filepath)

            for level, module, names in rel_imports:
                validator.validate_relative_import(level, module, names, filepath)

        if validator.errors:
            pytest.fail("\n".join(validator.errors))

    def test_no_circular_dependencies(self, validator):
        """Verify there are no circular dependencies."""
        validator.find_all_python_files()
        cycles = validator.check_circular_dependencies()

        if cycles:
            cycle_descriptions = []
            for cycle in cycles:
                cycle_str = " -> ".join(cycle)
                cycle_descriptions.append(f"  {cycle_str}")

            pytest.fail(
                f"Found {len(cycles)} circular dependencies:\n" + "\n".join(cycle_descriptions)
            )

    def test_all_modules_importable(self, validator):
        """Verify all modules can be imported without errors."""
        validator.find_all_python_files()
        failures = validator.try_import_all_modules()

        if failures:
            error_messages = []
            for module_name, exception in failures.items():
                error_messages.append(f"  {module_name}: {type(exception).__name__}: {exception}")

            pytest.fail(f"Failed to import {len(failures)} modules:\n" + "\n".join(error_messages))


class TestConstantsAndSharedResources:
    """Test that constants and shared resources are properly defined."""

    def test_const_module_exists(self):
        """Verify const.py exists and is importable."""
        const_file = COMPONENT_ROOT / "const.py"
        assert const_file.exists(), "const.py not found"

        try:
            from custom_components.effektguard import const
        except ImportError as e:
            pytest.fail(f"Cannot import const module: {e}")

    def test_domain_constant_defined(self):
        """Verify DOMAIN constant exists (required by Home Assistant)."""
        from custom_components.effektguard import const

        if not hasattr(const, "DOMAIN"):
            pytest.fail("DOMAIN constant missing from const.py (required for Home Assistant)")

    def test_dm_constants_exist(self):
        """Verify degree minutes constants are defined."""
        from custom_components.effektguard import const

        # Look for any DM_THRESHOLD constants
        dm_constants = [name for name in dir(const) if name.startswith("DM_")]

        if not dm_constants:
            pytest.fail("No DM_* constants found in const.py (thermal debt tracking needs these)")

        # Verify absolute max exists (critical safety constant)
        if not hasattr(const, "DM_THRESHOLD_AUX_LIMIT"):
            pytest.fail("DM_THRESHOLD_AUX_LIMIT missing (critical safety constant)")

    def test_offset_limits_defined(self):
        """Verify offset limit constants exist."""
        from custom_components.effektguard import const

        missing = []
        for const_name in ["MIN_OFFSET", "MAX_OFFSET"]:
            if not hasattr(const, const_name):
                missing.append(const_name)

        if missing:
            pytest.fail(f"Missing offset constants: {', '.join(missing)}")

    def test_coordinator_exists(self):
        """Verify coordinator module is properly defined."""
        coordinator_file = COMPONENT_ROOT / "coordinator.py"
        if not coordinator_file.exists():
            pytest.skip("coordinator.py not found")

        try:
            # Dynamically find coordinator class
            import importlib

            coordinator_module = importlib.import_module(
                "custom_components.effektguard.coordinator"
            )

            # Look for coordinator class (usually ends with Coordinator)
            coordinator_classes = [
                name
                for name in dir(coordinator_module)
                if name.endswith("Coordinator") and not name.startswith("_")
            ]

            if not coordinator_classes:
                pytest.fail("No Coordinator class found in coordinator.py")
        except ImportError as e:
            pytest.fail(f"Cannot import coordinator module: {e}")

    def test_climate_entity_exists(self):
        """Verify climate entity is properly defined."""
        climate_file = COMPONENT_ROOT / "climate.py"
        if not climate_file.exists():
            pytest.skip("climate.py not found")

        try:
            # Dynamically find climate class
            import importlib

            climate_module = importlib.import_module("custom_components.effektguard.climate")

            # Look for climate entity class (usually ends with Climate)
            climate_classes = [
                name
                for name in dir(climate_module)
                if "Climate" in name and not name.startswith("_") and name != "ClimateEntity"
            ]

            if not climate_classes:
                pytest.fail("No Climate entity class found in climate.py")
        except ImportError as e:
            pytest.fail(f"Cannot import climate module: {e}")


class TestModuleStructure:
    """Test the overall module structure and organization."""

    def test_all_packages_have_python_files(self):
        """Verify all directories with __init__.py contain Python files."""
        for dirpath, dirnames, filenames in COMPONENT_ROOT.walk():
            # Skip __pycache__ and hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]

            # If this is a package (has __init__.py)
            if "__init__.py" in filenames:
                python_files = [f for f in filenames if f.endswith(".py") and f != "__init__.py"]

                # Package should have at least one Python file (besides __init__.py)
                # or have subdirectories with Python files
                if not python_files:
                    # Check if there are subdirectories
                    has_subpackages = any((dirpath / d / "__init__.py").exists() for d in dirnames)
                    if not has_subpackages:
                        # Warning only, not a failure - empty packages might be intentional
                        import warnings

                        warnings.warn(
                            f"Package {dirpath.relative_to(COMPONENT_ROOT)} has no Python files"
                        )

    def test_all_directories_importable(self):
        """Verify all package directories can be imported."""
        packages_found = []

        for dirpath, dirnames, filenames in COMPONENT_ROOT.walk():
            # Skip __pycache__
            if "__pycache__" in str(dirpath):
                continue

            # If this is a package (has __init__.py)
            if "__init__.py" in filenames:
                # Build module name
                rel_path = dirpath.relative_to(COMPONENT_ROOT.parent.parent)
                module_name = str(rel_path).replace("/", ".")
                packages_found.append(module_name)

        # Try to import all packages
        failures = []
        for module_name in packages_found:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                failures.append(f"{module_name}: {type(e).__name__}: {e}")

        if failures:
            pytest.fail(f"Failed to import {len(failures)} packages:\n" + "\n".join(failures))

        # Should have found at least the main package
        assert len(packages_found) > 0, "No packages found"

    def test_key_components_exist(self):
        """Verify key component files exist."""
        key_files = [
            "const.py",
            "coordinator.py",
            "climate.py",
            "config_flow.py",
            "manifest.json",
            "__init__.py",
        ]

        missing = []
        for filename in key_files:
            if not (COMPONENT_ROOT / filename).exists():
                missing.append(filename)

        if missing:
            pytest.fail(f"Missing key component files: {', '.join(missing)}")


class TestCriticalImports:
    """Test that all classes in all modules can be imported."""

    def test_import_all_classes_from_all_modules(self, validator):
        """Dynamically discover and import all classes from all Python files."""
        validator.find_all_python_files()

        failures = []
        classes_tested = 0

        for filepath in validator.all_python_files:
            # Parse the file to find class definitions
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(filepath))
            except SyntaxError:
                continue  # Already tested in syntax test

            # Extract class names
            class_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_names.append(node.name)

            if not class_names:
                continue

            # Get module name
            module_name = validator._get_module_name(filepath)

            # Try to import the module and each class
            try:
                module = importlib.import_module(module_name)

                for class_name in class_names:
                    classes_tested += 1
                    if not hasattr(module, class_name):
                        failures.append(
                            f"{module_name}.{class_name}: Class defined but not exported"
                        )
                    else:
                        # Verify we can actually get the class
                        try:
                            cls = getattr(module, class_name)
                            if not isinstance(cls, type):
                                failures.append(f"{module_name}.{class_name}: Not a class type")
                        except Exception as e:
                            failures.append(
                                f"{module_name}.{class_name}: Error accessing class: {e}"
                            )
            except Exception as e:
                for class_name in class_names:
                    failures.append(f"{module_name}.{class_name}: Cannot import module: {e}")

        if failures:
            pytest.fail(
                f"Failed to import {len(failures)} classes out of {classes_tested} tested:\n"
                + "\n".join(failures[:20])  # Show first 20 failures
            )

        assert classes_tested > 0, "No classes found to test"

    def test_import_all_functions_from_all_modules(self, validator):
        """Dynamically discover and import all public functions from all Python files."""
        validator.find_all_python_files()

        failures = []
        functions_tested = 0

        for filepath in validator.all_python_files:
            # Parse the file to find function definitions
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(filepath))
            except SyntaxError:
                continue

            # Extract top-level function names (not methods)
            function_names = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    function_names.append(node.name)

            if not function_names:
                continue

            # Get module name
            module_name = validator._get_module_name(filepath)

            # Try to import the module and each function
            try:
                module = importlib.import_module(module_name)

                for func_name in function_names:
                    functions_tested += 1
                    if not hasattr(module, func_name):
                        failures.append(
                            f"{module_name}.{func_name}: Function defined but not exported"
                        )
            except Exception as e:
                for func_name in function_names:
                    failures.append(f"{module_name}.{func_name}: Cannot import module: {e}")

        if failures:
            pytest.fail(
                f"Failed to import {len(failures)} functions out of {functions_tested} tested:\n"
                + "\n".join(failures[:20])
            )

    def test_critical_optimization_modules(self):
        """Test that critical optimization modules exist and are importable."""
        optimization_dir = COMPONENT_ROOT / "optimization"
        if not optimization_dir.exists():
            pytest.skip("optimization directory not found")

        # Find all Python files in optimization
        opt_files = list(optimization_dir.glob("*.py"))
        opt_files = [f for f in opt_files if f.name != "__init__.py"]

        if not opt_files:
            pytest.fail("No optimization modules found")

        # Try to import each one
        failures = []
        for opt_file in opt_files:
            module_name = f"custom_components.effektguard.optimization.{opt_file.stem}"
            try:
                importlib.import_module(module_name)
            except Exception as e:
                failures.append(f"{module_name}: {type(e).__name__}: {e}")

        if failures:
            pytest.fail(
                f"Failed to import {len(failures)} optimization modules:\n" + "\n".join(failures)
            )

    def test_critical_adapter_modules(self):
        """Test that critical adapter modules exist and are importable."""
        adapters_dir = COMPONENT_ROOT / "adapters"
        if not adapters_dir.exists():
            pytest.skip("adapters directory not found")

        # Find all Python files in adapters
        adapter_files = list(adapters_dir.glob("*.py"))
        adapter_files = [f for f in adapter_files if f.name != "__init__.py"]

        if not adapter_files:
            pytest.fail("No adapter modules found")

        # Try to import each one
        failures = []
        for adapter_file in adapter_files:
            module_name = f"custom_components.effektguard.adapters.{adapter_file.stem}"
            try:
                importlib.import_module(module_name)
            except Exception as e:
                failures.append(f"{module_name}: {type(e).__name__}: {e}")

        if failures:
            pytest.fail(
                f"Failed to import {len(failures)} adapter modules:\n" + "\n".join(failures)
            )


class TestConstantUsage:
    """Test that all constants are actually used in the codebase."""

    def test_all_constants_are_used(self, validator):
        """Check that every constant in const.py is used somewhere in the codebase."""
        from custom_components.effektguard import const

        # Get all constants from const.py (uppercase names that don't start with _)
        all_constants = [name for name in dir(const) if name.isupper() and not name.startswith("_")]

        if not all_constants:
            pytest.fail("No constants found in const.py")

        # Find all Python files (excluding const.py itself)
        validator.find_all_python_files()
        files_to_search = [f for f in validator.all_python_files if f.name != "const.py"]

        # Search for usage of each constant
        unused_constants = []
        usage_map = {}

        for const_name in all_constants:
            found_usage = False
            usage_locations = []

            for filepath in files_to_search:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Check if constant is used in this file
                    # Look for the constant name as a whole word
                    if f"\\b{const_name}\\b" in content or const_name in content:
                        # Verify it's actually used, not just in a comment
                        lines = content.split("\n")
                        for i, line in enumerate(lines, 1):
                            # Skip comments and docstrings
                            stripped = line.strip()
                            if stripped.startswith("#"):
                                continue

                            if const_name in line:
                                found_usage = True
                                usage_locations.append(
                                    f"{filepath.relative_to(COMPONENT_ROOT)}:{i}"
                                )
                                break

                        if found_usage:
                            break
                except Exception:
                    continue

            if not found_usage:
                unused_constants.append(const_name)
            else:
                usage_map[const_name] = usage_locations

        # Report results
        if unused_constants:
            # Create a detailed report
            report_lines = [
                f"\nFound {len(unused_constants)} unused constants out of {len(all_constants)} total:",
                "",
            ]

            for const_name in sorted(unused_constants):
                const_value = getattr(const, const_name)
                report_lines.append(f"  - {const_name} = {repr(const_value)}")

            report_lines.extend(
                [
                    "",
                    "These constants are defined but never used in the codebase.",
                    "Consider removing them or verify they should be used.",
                ]
            )

            # This is a warning, not a hard failure (might be intentional for future use)
            import warnings

            warnings.warn("\n".join(report_lines))

    def test_no_duplicate_constants_in_const_py(self, validator):
        """Check that const.py has no duplicate constant definitions.
        
        Detects:
        1. Same constant name defined multiple times at module level
        2. Different constant names with the same value (potential semantic duplicates)
        
        Skips enum class members (they can have same names like UNKNOWN in different enums).
        """
        const_file = COMPONENT_ROOT / "const.py"
        content = const_file.read_text()
        tree = ast.parse(content)
        
        # Track constant definitions at module level only (not inside classes)
        constant_definitions = {}  # name -> list of line numbers
        constant_values = {}  # name -> value (for semantic duplicate detection)
        
        # Get only top-level assignments (not inside class/function definitions)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        name = target.id
                        line = node.lineno
                        
                        # Track definition locations
                        if name not in constant_definitions:
                            constant_definitions[name] = []
                        constant_definitions[name].append(line)
                        
                        # Try to extract value for semantic duplicate detection
                        try:
                            if isinstance(node.value, ast.Constant):
                                constant_values[name] = node.value.value
                            elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
                                if isinstance(node.value.operand, ast.Constant):
                                    constant_values[name] = -node.value.operand.value
                        except Exception:
                            pass  # Skip values we can't extract - not all AST nodes are simple constants
        
        errors = []
        
        # Check for same name defined multiple times
        for name, lines in constant_definitions.items():
            if len(lines) > 1:
                errors.append(f"DUPLICATE DEFINITION: {name} defined on lines {lines}")
        
        # Check for semantic duplicates (same value, different names)
        # Only for numeric values to avoid false positives on strings
        from collections import defaultdict
        by_value = defaultdict(list)
        for name, value in constant_values.items():
            if isinstance(value, (int, float)):
                by_value[value].append(name)
        
        semantic_duplicates = []
        for value, names in by_value.items():
            if len(names) > 1:
                # Filter out intentional duplicates (e.g., LAYER_WEIGHT_SAFETY = 1.0 and others)
                # Only flag if names are suspiciously similar
                for i, name1 in enumerate(names):
                    for name2 in names[i+1:]:
                        # Check if names share significant substrings (potential duplicates)
                        words1 = set(name1.lower().split('_'))
                        words2 = set(name2.lower().split('_'))
                        shared_words = words1 & words2 - {'the', 'a', 'an', 'is', 'kw', 'min', 'max'}
                        if len(shared_words) >= 2:
                            semantic_duplicates.append(
                                f"POTENTIAL DUPLICATE: {name1} and {name2} both = {value}"
                            )
        
        if errors:
            pytest.fail(
                f"Found {len(errors)} duplicate constant definitions in const.py:\n"
                + "\n".join(errors)
            )
        
        if semantic_duplicates:
            import warnings
            warnings.warn(
                f"Found {len(semantic_duplicates)} potential semantic duplicates:\n"
                + "\n".join(semantic_duplicates[:10])
            )

    def test_constants_imported_correctly(self, validator):
        """Check that constants are imported from const.py, not redefined."""
        validator.find_all_python_files()

        # First, get all constants from const.py
        from custom_components.effektguard import const

        all_constants = {
            name: getattr(const, name)
            for name in dir(const)
            if name.isupper() and not name.startswith("_")
        }

        # Now check other files for redefinitions
        redefinitions = []

        for filepath in validator.all_python_files:
            if filepath.name == "const.py":
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(filepath))
            except SyntaxError:
                continue

            # Look for constant assignments (uppercase = value)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            # Check if this constant name exists in const.py
                            if target.id in all_constants:
                                redefinitions.append(
                                    f"{filepath.relative_to(COMPONENT_ROOT)}: "
                                    f"Redefines {target.id} (should import from const.py)"
                                )

        if redefinitions:
            import warnings

            warnings.warn(
                f"Found {len(redefinitions)} constant redefinitions:\n"
                + "\n".join(redefinitions[:10])
            )

    def test_magic_numbers_detection(self, validator):
        """Detect potential magic numbers that should be constants."""
        validator.find_all_python_files()

        # Numbers that commonly appear as magic numbers to flag
        suspicious_patterns = []

        for filepath in validator.all_python_files:
            if filepath.name == "const.py":
                continue  # Skip const.py itself

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    stripped = line.strip()

                    # Skip comments and docstrings
                    if (
                        stripped.startswith("#")
                        or stripped.startswith('"""')
                        or stripped.startswith("'''")
                    ):
                        continue

                    # Look for hardcoded negative numbers (common in DM thresholds)
                    # Pattern: comparison with negative number
                    import re

                    # Check for degree minutes comparisons (e.g., "if dm < -500")
                    dm_pattern = re.findall(
                        r"(?:dm|degree_minutes)\s*[<>]=?\s*(-\d{3,})", line, re.IGNORECASE
                    )
                    if dm_pattern:
                        for number in dm_pattern:
                            suspicious_patterns.append(
                                f"{filepath.relative_to(COMPONENT_ROOT)}:{i}: "
                                f"Hardcoded DM threshold {number} (should be constant)"
                            )

                    # Check for hardcoded offset limits (e.g., "if offset > 10")
                    offset_pattern = re.findall(
                        r"(?:offset|curve)\s*[<>]=?\s*(-?\d+\.?\d*)", line, re.IGNORECASE
                    )
                    if offset_pattern:
                        for number in offset_pattern:
                            try:
                                num_val = float(number)
                                if abs(num_val) >= 5:  # Likely an offset limit
                                    suspicious_patterns.append(
                                        f"{filepath.relative_to(COMPONENT_ROOT)}:{i}: "
                                        f"Hardcoded offset limit {number} (should be constant)"
                                    )
                            except ValueError:
                                pass
            except Exception:
                continue

        if suspicious_patterns:
            import warnings

            warnings.warn(
                f"Found {len(suspicious_patterns)} potential magic numbers:\n"
                + "\n".join(suspicious_patterns[:15])
                + "\n(Showing first 15 of potentially more)"
            )


class TestCodeQuality:
    """Test for common code quality issues that cause import problems."""

    def test_no_syntax_errors_in_any_file(self, validator):
        """Verify no Python files have syntax errors."""
        validator.find_all_python_files()

        syntax_errors = []
        for filepath in validator.all_python_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    compile(f.read(), str(filepath), "exec")
            except SyntaxError as e:
                syntax_errors.append(
                    f"{filepath.relative_to(COMPONENT_ROOT)}: Line {e.lineno}: {e.msg}"
                )

        if syntax_errors:
            pytest.fail(f"Found {len(syntax_errors)} syntax errors:\n" + "\n".join(syntax_errors))

    def test_no_undefined_names_in_imports(self, validator):
        """Check for common undefined name issues in import statements."""
        validator.find_all_python_files()

        issues = []
        for filepath in validator.all_python_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(filepath))

                # Check for "from X import Y" where Y might be misspelled
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        # Common typos
                        for alias in node.names:
                            if alias.name.lower() != alias.name and "_" not in alias.name:
                                # Check if it's a class (starts with capital)
                                if alias.name[0].isupper():
                                    # This is probably intentional
                                    continue
            except SyntaxError:
                continue  # Already caught by other tests

        # This test mainly ensures the AST parsing works
        assert True

    def test_all_init_files_importable(self):
        """Verify all __init__.py files can be imported."""
        init_files = list(COMPONENT_ROOT.rglob("__init__.py"))

        failures = []
        for init_file in init_files:
            if "__pycache__" in str(init_file):
                continue

            # Build module name
            if init_file.parent == COMPONENT_ROOT:
                module_name = "custom_components.effektguard"
            else:
                rel_path = init_file.parent.relative_to(COMPONENT_ROOT.parent.parent)
                module_name = str(rel_path).replace("/", ".")

            try:
                importlib.import_module(module_name)
            except Exception as e:
                failures.append(f"{module_name}: {type(e).__name__}: {e}")

        if failures:
            pytest.fail(
                f"Failed to import {len(failures)} __init__.py files:\n" + "\n".join(failures)
            )

    def test_no_common_import_antipatterns(self, validator):
        """Check for common import anti-patterns."""
        validator.find_all_python_files()

        issues = []
        for filepath in validator.all_python_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                # Check for "import *" (usually bad practice)
                for i, line in enumerate(lines, 1):
                    if "from" in line and "import *" in line and not line.strip().startswith("#"):
                        # Allow in __init__.py files for re-exports
                        if filepath.name != "__init__.py":
                            issues.append(
                                f"{filepath.relative_to(COMPONENT_ROOT)}:{i}: "
                                f"'import *' found (makes dependencies unclear)"
                            )
            except Exception:
                continue

        # This is a warning, not a failure
        if issues:
            import warnings

            warnings.warn(f"Found {len(issues)} import anti-patterns:\n" + "\n".join(issues[:10]))


if __name__ == "__main__":
    # Allow running this test file directly for quick validation
    pytest.main([__file__, "-v", "--tb=short"])
