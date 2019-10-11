from pybuilder.core import use_plugin, init

use_plugin("python.core")
use_plugin('pypi:pybuilder_pytest')
use_plugin("python.install_dependencies")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin('python.sphinx')
use_plugin('python.integrationtest')
use_plugin("exec")


name = "mrcnn"
default_task = "publish"


@init
def set_properties(project):
    project.plugin_depends_on("flake8", "~=3.5")
    project.depends_on_requirements("requirements.txt")
    project.build_depends_on('mockito')
    project.set_property_if_unset("filter_resources_target", "$dir_target")
    project.set_property_if_unset("flake8_break_build", False)
    project.set_property_if_unset("flake8_max_line_length", 120)
    project.set_property_if_unset("flake8_include_patterns", None)
    project.set_property_if_unset("flake8_exclude_patterns", None)
    project.set_property_if_unset("flake8_include_test_sources", False)
    project.set_property_if_unset("flake8_include_scripts", True)
    project.set_property_if_unset("flake8_max_complexity", None)
    project.set_property_if_unset("flake8_verbose_output", False)
    project.get_property("pytest_extra_args").append("-v")
    project.get_property("pytest_extra_args").append("-x")
    project.set_property("coverage_break_build", False) # Change this to true to stop build if coverage is low
    project.set_property("sphinx_config_path", "docs/source/")
    project.set_property("sphinx_source_dir", "docs/source/")
    project.set_property("sphinx_output_dir", "docs/_build")
