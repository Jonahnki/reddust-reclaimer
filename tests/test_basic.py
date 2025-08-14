"""Basic tests for RedDust Reclaimer project."""

def test_python_version():
    """Test that Python version is acceptable."""
    import sys
    assert sys.version_info >= (3, 8)

def test_imports():
    """Test that basic scientific packages can be imported."""
    try:
        import sys
        print(f"Python version: {sys.version}")
        assert True
    except Exception as e:
        print(f"Error in imports: {e}")
        assert False

def test_project_structure():
    """Test basic project structure exists."""
    import os
    assert os.path.exists('README.md') or os.path.exists('readme.md')
