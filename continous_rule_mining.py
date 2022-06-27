# Copyright (C) 2022  Beate Scheibel
# This file is part of dmma.
#
# dmma is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# dmma is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# dmma.  If not, see <http://www.gnu.org/licenses/>.


import pandas as pd
import numpy as np
np.seterr(divide='ignore')
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support as score
from csv import DictReader
import collections
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.tree import export_text  # tree diagram


def slice_odict(odict, start=None, end=None):
    return collections.OrderedDict([
        (k, v) for (k, v) in odict.items()
        if k in list(odict.keys())[int(start):end]
    ])


def create_trace_dict_bpic20(row, trace_dict, id, result, interest, end_results):
    id = row[id]
    if id in trace_dict.keys():
        trace_dict[id]["events"].append(row["concept:name"])  # append(row["event"])
        for v in interest:
            if row[v] != "":
                values = row[v]
                trace_dict[id][v] = float(values)
        if row[result] != "":
            trace_dict[id][result] = row[result]
        if row["concept:name"] in end_results:  # row["event"]
            trace_dict[id]["last_timestamp"] = row["time:timestamp"]  #
            return True
    else:
        trace_dict[id] = {}
        trace_dict[id]["first_timestamp"] = row["time:timestamp"]  # row["timestamp"]
        trace_dict[id]["events"] = []
        trace_dict[id]["events"].append(row["case:concept:name"])  # [row["event"]]
    return False

def create_trace_dict(row, trace_dict, id, result, interest, end_results):
    id = row[id]
    if id in trace_dict.keys():
        trace_dict[id]["events"].append(row["event"]) #append(row["event"])
        for v in interest:
            if row[v] != "":
                values = row[v]
                trace_dict[id][v] = float(values)
        if row[result] != "":
            trace_dict[id][result] = row[result]
        if row["event"] in end_results: #row["event"]
            trace_dict[id]["last_timestamp"] = row["timestamp"] #
            return True
    else:
        trace_dict[id] = {}
        trace_dict[id]["first_timestamp"] =  row["timestamp"] #row["timestamp"]
        trace_dict[id]["events"] = []
        trace_dict[id]["events"].append(row["event"]) #[row["event"]]
    return False


def print_history(clf, X_var, counter_end, mean_accuracy, use_case, mean_fscore):
    global old_rules
    rules = (export_text(clf, list(X_var)))
    if rules == old_rules:
        return False
    old_rules = rules
    f = open("rule_history" + use_case + ".txt", mode="a")
    f.write("Mean accuracy: " + str(mean_accuracy) + "\n")
    f.write("Mean F1 Score: " + str(mean_fscore) + "\n")

    f.write("-" * 100 + "\n")
    f.write("From instance " + str(counter_end) + "\n")
    f.write(rules)
    f.write("\n")
    f.close()
    return True


def hoeffding_tree(df, result_column, names, counter_end, mean_accuracy, use_case, mean_fscore):
    y_var = df[result_column].values
    X_var = df[names]
    clf = tree.DecisionTreeClassifier(random_state=0, max_depth=4)
    y_var = np.where(y_var==0, 'False', y_var) #for bpi2020
    model = clf.fit(X_var, y_var)
    print_history(model, X_var, counter_end, mean_accuracy, use_case,mean_fscore)

    return model


def mine_rules(trace_dict, result_column, id, counter_end, mean_accuracy, use_case, mean_fscore):
    df = pd.DataFrame.from_dict(trace_dict, orient="index")
    try:
        df = df.rename_axis(id).reset_index()
    except:
        df = df.dropna().reset_index()
    if use_case=="permit_bpi20":
        df = df.fillna(0).dropna(axis=1).reset_index()
    names = df.select_dtypes(include=np.number).columns.tolist()
    names = [n for n in names if n != result_column and n!=id and n!="index"]
    model = hoeffding_tree(df, result_column, names, counter_end, mean_accuracy, use_case, mean_fscore)
    X_test = df.loc[:, names]
    y_test = df[result_column]
    if use_case== "permit_bpi20":
        y_test = np.where(y_test == 0, 'False', y_test)  # for bpi2020
        labels = np.unique(y_test) #bpi2020
    else:
        labels = y_test.unique()
    pred_model = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_model)
    precision, recall, fscore, support = score(y_test, pred_model, average="weighted")
    lgloss = log_loss(y_test, model.predict_proba(X_test), labels=labels)
    return df, model, accuracy, lgloss, fscore

def check_accuracy(model, trace_dict, variable_interest, result, id, use_case):
    df = pd.DataFrame.from_dict(trace_dict, orient="index")
    try:
        df = df.rename_axis(id).reset_index()
        if use_case=="permit_bpi20":
            df = df.fillna(0).dropna(axis=1).reset_index()  # bpic2020
    except:
        pass
    names = df.select_dtypes(include=np.number).columns.tolist()
    names = [n for n in names if n != result and n != id and n!="index"]
    X_test = df.loc[:, names]
    n_features = (model.n_features_)
    y_test = df[result]
    if use_case == "permit_bpi20":
        labels = np.unique(y_test) #bpi2020
    else:
        labels = y_test.unique()

    pred_model = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_model)
    precision, recall, fscore, support = score(y_test, pred_model, average="weighted")
    node_counts = model.tree_.node_count
    depth = model.tree_.max_depth
    complexity = node_counts + depth
    lgloss = log_loss(y_test, model.predict_proba(X_test), labels=labels)
    return lgloss, accuracy, complexity, fscore


def main(use_case):
    trace_dict = collections.OrderedDict()
    rule_model = None
    initial = True
    if use_case == "loan":
        file = "data/loan.csv"
        result_column = "check"
        id = "uuid"
        end_results = "Inform Customer"
    elif use_case == "seasonal":
        file = "data/seasonal_scenario.csv"
        result_column = "price_category"  # "result"
        id = "uuid"
        end_results = ["Send offer"]
    elif use_case=="added_sensor":
        file = "data/added_sensor.csv"
        result_column = "check"  # "result"
        id = "uuid"
        end_results = ["Put Workpiece in OK Pile", "Put Workpiece in Scrap Pile"]
    elif use_case == "permit_bpi20":
        file = "data/permit_all.csv"
        result_column = "case:Overspent"  # "result"
        id = "case:concept:name"
        end_results = ["Payment Handled"]

    f = open("rule_history"+use_case+".txt", mode="w")
    f.write("")
    f.truncate()
    f.close()
    counter = 0
    counter_end = 0
    counter_remine = 0
    counter_total_accuracy = 1
    total_accuracy = 0
    grace_period = 200
    grace_period_small = grace_period
    grace_period_larger = grace_period
    unsure = False
    remine = False
    total_fscore = 0
    max_grace_period = 1000
    min_grace_period = 100

    with open(file, 'r') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            variable_interest = []

            for key in row:
                if row[key].replace('.', '', 1).isnumeric():
                    variable_interest.append(key)
            if use_case == "permit_bpi20":
                variable_interest = ['case:OverspentAmount']
                first_finished = create_trace_dict_bpic20(row, trace_dict, id, result_column, variable_interest, end_results)
            else:
                first_finished = create_trace_dict(row, trace_dict, id, result_column, variable_interest, end_results)

            if first_finished:
                counter_end += 1
                counter += 1
                counter_remine += 1

                if grace_period > max_grace_period:
                    grace_period = max_grace_period
                if grace_period < min_grace_period:
                    grace_period = min_grace_period
                if grace_period_small < min_grace_period:
                    grace_period = min_grace_period
                if counter_end < grace_period:
                    continue
                if initial:
                    df, rule_model, accuracy_score, log_loss, fscore = mine_rules(trace_dict, result_column, id, int(counter_end),
                                                                total_accuracy, use_case,
                                                                                  total_fscore)
                    total_accuracy = accuracy_score
                    counter = 0
                    total_fscore = fscore
                    initial = False
                if (not unsure and counter_remine > grace_period) or unsure and (
                        counter_remine > grace_period_larger or counter_remine > grace_period_small):
                    if unsure and counter_remine > grace_period_larger:
                        set_grace_period = grace_period_larger
                    elif unsure:
                        set_grace_period = grace_period_small
                    else:
                        set_grace_period = grace_period
                    counter_remine = 0
                    try:
                        fscore_old = fscore
                        log_loss, accuracy, complexity, fscore = check_accuracy(rule_model, slice_odict(trace_dict,
                                                                                                        start=-set_grace_period),
                                                                                variable_interest, result_column,
                                                                                id, use_case)
                        total_accuracy += accuracy
                        total_fscore += fscore
                        print("fscore new:", fscore)
                    except Exception as e:
                        remine = True
                    if (fscore < 0.9 * fscore_old) or remine:
                        counter_total_accuracy += 1
                        df, rule_model, accuracy_score, log_loss, fscore = mine_rules(slice_odict(trace_dict,
                                                                                                  start=-(
                                                                                                      set_grace_period)),
                                                                                      result_column, id,
                                                                                      int(counter_end),
                                                                                      total_accuracy / counter_total_accuracy,
                                                                                      use_case,
                                                                                      total_fscore / counter_total_accuracy)
                        counter_total_accuracy = 1
                        total_accuracy = accuracy_score
                        counter = 0
                        total_fscore = fscore
                        remine = False
                    else:
                        counter_total_accuracy += 1
                        if fscore > 0.98:
                            grace_period *= 1.01
                            unsure = False
                            grace_period_small = set_grace_period
                            grace_period_larger = set_grace_period
                        elif fscore <= 0.90:
                            unsure = True
                            grace_period_small *= 0.8
                            grace_period_larger *= 1.2

            if len(trace_dict)> grace_period_larger:
                trace_dict.popitem(last=False)

    f = open("rule_history"+ use_case +".txt", mode="a")
    if counter_total_accuracy == 0:
        counter_total_accuracy = 1
    f.write("Last Instance: " + str(counter_end) + "\n")
    f.write("Mean accuracy: " + str(total_accuracy/ counter_total_accuracy) + "\n")
    f.write("Mean F1 Score: " + str(total_fscore / counter_total_accuracy) + "\n")

    f.close()

old_rules = ""


if __name__ == "__main__":

    print("added sensor")
    main(use_case="added_sensor")
    print("seasonal")
    main(use_case="seasonal")
    print("loan")
    main(use_case="loan")
    #print("bpi20")
    #main(use_case="permit_bpi20")
