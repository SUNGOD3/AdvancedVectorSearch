import os
import sys
import coverage
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def run_coverage():
    cov = coverage.Coverage(
        source=[os.path.join(project_root, 'src')], 
        omit=['*test*', '*setup.py'],
    )

    cov.start()

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.join(project_root, 'tests'))
    test_runner = unittest.TextTestRunner()
    result = test_runner.run(test_suite)

    cov.stop()

    print("\n--- Coverage Report ---")
    cov.report()

    cov.html_report(directory=os.path.join(project_root, 'coverage_html'))
    
    cov.xml_report(outfile=os.path.join(project_root, 'coverage.xml'))

    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    run_coverage()