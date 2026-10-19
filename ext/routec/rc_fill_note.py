"""fill the {{...}} placeholders of the scoping note from scoping_tables.json (generated from raw records)."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, 'scoping_tables.json')))
p = os.path.join(HERE, 'ROUTE_C_SCOPING_2026-10-18.md'); s = open(p).read()
gc = T['GOAL_CHECKS']
rep = {'{{GOAL}}': T['GOAL'], '{{JOINT}}': T['JOINT'], '{{CONTRAST}}': T['CONTRAST'], '{{INDEX}}': T['INDEX'],
       '{{GLOBAL}}': T['GLOBAL'], '{{LIFTFREE}}': T['LIFTFREE'], '{{STRESS}}': T['STRESS'],
       '{{GOAL_N}}': str(gc['n']), '{{GOAL_MOM}}': f"{gc['max_moment_defect']:.1e} (relative)", '{{GOAL_DIV}}': f"{gc['max_div_defect']:.1e}",
       '{{J5_ch}}': f"{gc['ref']['channels']['J5']:.4e}", '{{r5_ch}}': f"{gc['ref']['channels']['r5']:.2e}",
       '{{J5_ba}}': f"{gc['ref']['barriers']['J5']:.4e}", '{{r5_ba}}': f"{gc['ref']['barriers']['r5']:.2e}"}
for k, v in rep.items():
    if k not in s: print('missing placeholder', k)
    s = s.replace(k, v)
open(p, 'w').write(s); print('filled; remaining placeholders:', s.count('{{'))
