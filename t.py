
def foo(fvs, l_key_attr, r_key_attr):
    feature_attrs = list(fvs.columns)
    feature_attrs.remove(l_key_attr)
    feature_attrs.remove(r_key_attr)

    ub = {}
    for k in feature_attrs:
        ub[k] = 1.0
    proj_fvs = fvs[feature_attrs]
    n=len(feature_attrs)
    for row in proj_fvs.itertuples(index=False):
        for i in xrange(n):
            if row[i] > ub[feature_attrs[i]]:
                ub[feature_attrs[i]] = row[i]
                print(feature_attrs[i], row[i])

    f = open('upper_bound1', 'w')
    for k in ub.keys():
        f.write(k + ',' + str(ub[k]) + '\n')
    f.close()        
