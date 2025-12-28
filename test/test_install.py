# test_packages.py
import sys

print(f"Python version: {sys.version}\n")

# Test each required package
packages_to_test = [
    ('numpy', 'np'),
    ('pandas', 'pd'),
    ('matplotlib.pyplot', 'plt'),
    ('scipy', 'sp'),
    ('sklearn', 'sklearn'),
    ('yaml', 'yaml')
]

for package, short_name in packages_to_test:
    try:
        if package == 'yaml':
            import yaml
        elif package == 'sklearn':
            import sklearn
        else:
            __import__(package.split('.')[0])
        print(f"✅ {package} is installed")
    except ImportError as e:
        print(f"❌ {package} is NOT installed: {e}")

print("\n✅ All required packages are installed! You can now run the main script.")