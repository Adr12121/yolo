import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Maintenant run le module
import runpy
runpy.run_path('repertoire_lookup.py', run_name='__main__')
