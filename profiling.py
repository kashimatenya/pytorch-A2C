import cProfile
import pstats

#get time profile with cProfile
script_name = "train_pendulum"
profile_name = "train_pendulum.prof"

cProfile.run("import "+script_name, profile_name)
stats = pstats.Stats(profile_name)
stats.sort_stats('tottime')
stats.print_stats(10)
