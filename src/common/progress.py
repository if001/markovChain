def progress(progress, max_num):
    sys.stdout.write("\r progress: %d / %d" % (progress, max_num))
    sys.stdout.flush()
