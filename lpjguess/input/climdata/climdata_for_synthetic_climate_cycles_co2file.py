import sys

for val in [180, 400]:

    out_lines= []

    for i in range(35000):
        out = "%d %.1f\n" % (i, val)
        out_lines.append(out)

    open('co2_TraCE_egu2018_35ka_const%dppm.txt' % val, 'w').write(''.join( out_lines ) )


