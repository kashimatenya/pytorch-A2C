import datetime
from collections import namedtuple
from time import sleep

#log data
def aggregate_stats(epoch, cumulative_step, test_stats, training_stats):
    return {
                "epoch": epoch, 
                "cumulative_step":cumulative_step,
                "test_steps":test_stats["test_steps"]["mean"], 
                "best_cumulative_reward":test_stats["cumulative_reward"]["best"], 
                "mean_cumulative_reward":test_stats["cumulative_reward"]["mean"], 
                "worst_cumulative_reward":test_stats["cumulative_reward"]["worst"],
                "training_V_loss": training_stats["training_loss"]["V"]["loss"], 
                "training_pi_loss": training_stats["training_loss"]["pi"]["loss"], 
                "training_entropy": training_stats["training_loss"]["pi"]["entropy"], 
                "test_V_loss" :test_stats["test_loss"]["V"]["loss"],
                "test_pi_loss" :test_stats["test_loss"]["pi"]["loss"],
                "test_pi_entropy" :test_stats["test_loss"]["pi"]["entropy"],
                "learning_rate" : training_stats["learning_rate"]
            }
#def


#print log to display
class DisplayLog:
    def __int__(self):
        pass
    #def

    def print(self, record):
        print( "epoch:" +str(record["epoch"])
              +", cumulative_step:" +str(record["cumulative_step"])
              +", test_steps:" +str(record["test_steps"])
              +", best_cumulative_reward:{:.2f}".format(record["best_cumulative_reward"])
              +", mean_cumulative_reward:{:.2f}".format(record["mean_cumulative_reward"])
              +", worst_cumulative_reward:{:.2f}".format(record["worst_cumulative_reward"])
            )
    #def
#class


#print log to file
class FileLog:
    def __init__(self, filename):
        self._retry = 0.1
        self._filename = filename

        self._initialzed = False

    #def

    def print(self, record):
        if not self._initialzed:
            self._initailze(record)
            self._initialzed = True
        #if

        #retry until successful
        while True:
            try:
                with open(self._filename, "a") as file:
                    line = list(map(str, record.values()))
                    line.append(str(datetime.datetime.now()))

                    file.write(','.join(line) +"\n")
                #with open(self._filename, "a") as file:

                break

            except Exception as e:
                sleep(self._retry)
                continue
            #try-except
        #while

        return
    #def


    def _initailze(self, record):
        with open(self._filename, "a") as file:
            file.write(",".join(record.keys())+"\n")

        return
    #def
#class


class Logs:
    def __init__(self, logs):
        self._logs = logs
    #def


    def print(self, data):
        for log in self._logs:
            log.print(data)

        return
    #def
#class
