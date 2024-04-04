# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:05:53 2024

@author: mhw20
"""
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import pickle

# CURRENTLY: The version we have in the cluster is using dates from 2011 to 2015.
sightings_location = "C:/dolphin_data/allsightings.csv"
orig_data = pd.read_csv(sightings_location)

# First, let's only consider sightings between a certain date.
lower_bound = "2010-06-01"
upper_bound = "2015-06-01"
data = orig_data[orig_data["date"] > lower_bound][orig_data["date"] < upper_bound]

# Next, consider infected sightings.
infected_sightings = data[data["infection_status"] == 1]
print("Total infected sightings: " + str(len(infected_sightings)))
noninfected_sightings = data[data["infection_status"] == 0]
print("Total non-infected sightings: " + str(len(noninfected_sightings)))

# Next, consider infected individuals.
infected_individuals = list(set(list(infected_sightings["dolphin_id"])))
print("Number of infected individuals: " + str(len(infected_individuals)))
# Next, let's see which of these individuals were infected before the lower bound date.
prev_data = orig_data[orig_data["date"] < lower_bound]
prev_infected_sightings = prev_data[prev_data["infection_status"]==1]
prev_infected_individuals = list(set(list(prev_infected_sightings["dolphin_id"])))
new_i = 0
infection_origin = []
prev_infection_times = []
for i in infected_individuals:
    if i not in prev_infected_individuals:
        new_i += 1
    else:
        infection_origin.append(i)
print("Newly infected during this period: " + str(new_i))
print("Source of infection: " + str(infection_origin))

"""
Our infection origin are the dolphins who were spotted with disease before and after the initial start date
"""
            


# Lastly, we can take a look at who recovered during this period.
num_recovered = 0
for i in infected_individuals:
    a = data[data["dolphin_id"] == i]
    max_sighting = max(a["date"])
    last_sighting_status = a[a["date"] == max_sighting]["infection_status"]
    if max(last_sighting_status) == 0: # Actually just have to have a "max" here since we can have multiple sightings on same day.
        num_recovered += 1
print("Number recovered: " + str(num_recovered))

# Find "birth dates" of dolphins, load in allfocals.
focals_location = "C:/dolphin_data/allfocals.csv"
focals_data = pd.read_csv(focals_location)
confirmed_born_before = 0
confirmed_born_during = 0
birthday_unknown = 0
born_before = 0
first_sightings = []
for i in infected_individuals:
    birthday = focals_data[focals_data["dolphin_id"] == i]["birthdate"]
    if len(focals_data[focals_data["dolphin_id"] == i]) == 0:
        birthday_unknown += 1
        first_sightings.append(min(orig_data[orig_data["dolphin_id"] == i]["date"]))
        # And check their earliest known sighting.
        if min(orig_data[orig_data["dolphin_id"] == i]["date"]) < lower_bound:
            born_before += 1
        continue
    #print(birthday.item())
    if str(birthday.item()) > lower_bound: # If dolphin was born during the study.
        confirmed_born_during += 1
    elif str(birthday.item()) > lower_bound <= lower_bound:
        confirmed_born_before += 1
print("Number of infected individuals confirmed to be born during study: " + str(confirmed_born_during))
print("Number of infected individuals confirmed to be born before study: " + str(confirmed_born_before))
print("Number of dolphins with unknown birthdays: " + str(birthday_unknown))
print("Number of unknowns sighted before sighting time: " + str(born_before))
print("First sightings of unknown birthdays: " + str(first_sightings))


total_individuals = list(set(list(data["dolphin_id"])))
print("Total individuals spotted during this period: "  + str(len(total_individuals)))

truncation_threshold = 12
"""
IMPORTANT: here, we basically ignore all dolphins who were NOT spotted at least twice, three months apart.
"""
granularity = 7
remaining_individuals = []
truncated_individuals = []
# Now deal with truncation.
# If the time between the first and last sighting of the dolphin is less than 4
for i in total_individuals:
    first_sighting = min(orig_data[orig_data["dolphin_id"] == i]["date"])
    last_sighting = max(orig_data[orig_data["dolphin_id"] == i]["date"])
    delta = datetime.date.fromisoformat(last_sighting) - datetime.date.fromisoformat(first_sighting)
    if int(delta.days)//int(granularity) > truncation_threshold:
        remaining_individuals.append(i)
    else:
        truncated_individuals.append(i)
total_individuals = remaining_individuals
print("In our dataset, " + str(truncated_individuals) + " were not being spotted enough.")
print(str(len(total_individuals)) + " remaining.")
# Next, let's make a huge aggregate network of every interaction ever.
node_keys = [] # Mapping IDs to nodes.
curr_node = 0
big_network = nx.Graph()
for i in total_individuals: # total_individuals is a list of all dolphin id's.
    node_keys.append(i)
    big_network.add_node(curr_node)
    curr_node += 1
num_nodes = len(node_keys)

    

        
        

cutoffs = ["2011-06-01", "2012-06-01", "2013-06-01", "2014-06-01", "2015-06-01"]
X = np.zeros(shape = (num_nodes, num_nodes)) # Matrix that will hold the observations observed between each dolphin.
X_dynamic = np.zeros(shape = (len(cutoffs), num_nodes, num_nodes)) # Matrix that holds the dynamic time blocks.
all_observations = list(set(list(data["observation_id"])))
for o in all_observations:
    obs_data = data[data["observation_id"] == o]
    spotted_individuals = list(set(list(obs_data["dolphin_id"])))
    combos = list(itertools.combinations(spotted_individuals,2))
    obs_date = list(data[data["observation_id"] == o]["date"])[0]
    curr_time_slot = 1000
    for i in range(len(cutoffs)):
        if obs_date < cutoffs[i]:
            curr_time_slot = i
            break
    for c in combos:
        if c[0] not in total_individuals or c[1] not in total_individuals:
            continue
        node_1 = node_keys.index(c[0])
        node_2 = node_keys.index(c[1])
        X[node_1, node_2] += 1
        X[node_2, node_1] += 1
        X_dynamic[curr_time_slot, node_1, node_2] += 1
        X_dynamic[curr_time_slot, node_2, node_1] += 1
network_transition_times = []
for c in cutoffs:
    delta = datetime.date.fromisoformat(c) - datetime.date.fromisoformat(lower_bound)
    network_transition_times.append(int(delta.days)//int(granularity))

test_times_dict = {i: [] for i in range(len(node_keys))}
results_dict = {i: [] for i in range(len(node_keys))}

for obs in range(len(data)):
    row = data.iloc[obs]
    day_delta = datetime.date.fromisoformat(data.iloc[obs]["date"]) - datetime.date.fromisoformat(lower_bound)
    week = int(day_delta.days)//int(granularity)
    if row["dolphin_id"] not in total_individuals:
        continue
    node = node_keys.index(row["dolphin_id"])
    if week in test_times_dict[node]:
        if row["infection_status"] > 0:
            results_dict[node][len(results_dict[node])-1] = 1
    else:
        test_times_dict[node].append(week)
        results_dict[node].append(row["infection_status"])
for i in infected_individuals:
    if i not in total_individuals:
        continue
    node = node_keys.index(i)
    print(results_dict[node])
total_time_delta = datetime.date.fromisoformat(upper_bound) - datetime.date.fromisoformat(lower_bound)
total_slots = int(total_time_delta.days)/int(granularity)

"""
See who is ineligible to be source of epidemic
Exclusion criterion for being a candidate:
    - Initially infected.
    - Infected in the span of 20 weeks around t= 0.
    
"""

non_candidates = []
pre_sightings = [] # The sighting right before t = 0
post_sightings = [] # The sighting right after t = 0
recovered_origin = []
infection_times = []
for d in total_individuals:
    a = orig_data[orig_data["dolphin_id"] == d]
    dates = list(a["date"])
    status = list(a["infection_status"])
    pre_sighting = ("NA", "NA")
    for i in range(len(dates)):
        if dates[i] <= lower_bound:
            pre_sighting = (status[i], dates[i])
        else: # If we exceed, then take that as the first post-t = 0 sighting.
            post_sighting = (status[i], dates[i])
            post_sightings.append(post_sighting)
            pre_sightings.append(pre_sighting)
            break
# At this point, see who is ineligible for being an original infected.
for i in range(len(total_individuals)):
    if total_individuals[i] in infected_individuals:
        if post_sightings[i][0] == 0: # If they were eventually infected but initially spotted clean, they're not infected at time 0.
            non_candidates.append(total_individuals[i])
        # Also call them non-candidates if we spot them as infected AFTER a year.
        elif (datetime.date.fromisoformat(post_sightings[i][1]) - datetime.date.fromisoformat(lower_bound)).days/7 > 2*54:
            non_candidates.append(total_individuals[i])
    # Initially infected individuals are not "candidates", since they should have already recovered. And those that were only infected before
    # can no longer be infected again.
    elif total_individuals[i] in prev_infected_individuals:
        non_candidates.append(total_individuals[i])
        recovered_origin.append(total_individuals[i])
    if pre_sightings[i][1] == "NA":
        continue
    time_lower = (datetime.date.fromisoformat(lower_bound) - datetime.date.fromisoformat(pre_sightings[i][1])).days/7
    time_upper = (datetime.date.fromisoformat(post_sightings[i][1]) - datetime.date.fromisoformat(lower_bound)).days/7
    # If we were spotted in the span of 20 weeks around t = 0 without an infection, then we are also not a candidate.
    if time_lower < 10 and time_upper < 10 and pre_sightings[i][0] == 0 and post_sightings[i][0] == 0:
        non_candidates.append(total_individuals[i])

birthdays = [0] * len(node_keys)
for i in range(len(node_keys)):
     dolphin = node_keys[i]
     birthday = focals_data[focals_data["dolphin_id"] == dolphin]["birthdate"]
     if len(birthday) > 0:
         birthday_delta = datetime.date.fromisoformat(str(birthday.item())) - datetime.date.fromisoformat(lower_bound)
         week = int(birthday_delta.days)//int(granularity)
         birthdays[i] = week
     else:
         birthdays[i] = -1 


non_candidates = list(set(non_candidates))
print(str(len(non_candidates)) + " are not eligible for initial infection.")
print(str(len(recovered_origin)) + " recovered already.")
"""
"""



index_infection_origin = []
for i in infection_origin:
    index_infection_origin.append(node_keys.index(i))
    
index_recovered_origin = []
for i in recovered_origin:
    index_recovered_origin.append(node_keys.index(i))

candidates = []
for i in total_individuals:
    if i not in non_candidates and i not in infection_origin:
        candidates.append(node_keys.index(i))


def find_first_time(dates, status, search_term):
    first_date = "NA"
    for i in range(len(status)):
        if status[i] == search_term:
            first_date = dates[i]
            break
    return first_date

# Now, look at the age statuses
ages_location = "C:/Users/mhw20/Documents/dolphin_analysis/dolphin_data/powell_aged_sightings.csv"
aged_data = pd.read_csv(ages_location)
#aged_data = aged_data[aged_data["date"] > lower_bound][aged_data["date"] < upper_bound]
def to_iso(time_str, ):
    return datetime.datetime.strptime(time_str, "%m/%d/%Y").strftime("%Y-%m-%d")
def time_diff(t1, t2, granularity):
    delta = datetime.date.fromisoformat(t2) - datetime.date.fromisoformat(t1)
    return int(delta.days)//int(granularity)


a_status_list = ["c1", "c2", "c3", "j", "a", "b"]
age_dict = {i: {j: -1000 for j in a_status_list} for i in range(len(node_keys))} # The time at which each individual transitions to their new state.
    
inconsistencies = 0 # Inconcsistencies occur if the date inferred by a birthday happens AFTER what we observe.
# Basically, this happens if a dolphin becomes a juvenile sooner than would be expected by the birthday.
for i in range(len(node_keys)):
    dolphin = node_keys[i]
    current_state = "unknown"
    a = aged_data[aged_data["dolphin_id"] == dolphin]
    
    # First, check if there's a birthday on record.
    birthday = focals_data[focals_data["dolphin_id"] == dolphin]["birthdate"]
    b_inferred = {s: -1 for s in a_status_list}
    found_birthday = False
    if len(birthday) > 0:
        found_birthday = True
        b_inferred["b"] = 1
        b_inferred["c1"] = time_diff(lower_bound, str(birthday.item()), granularity)
        b_inferred["c2"] = b_inferred["c1"] + 52
        b_inferred["c3"] = b_inferred["c1"] + 52 * 2
        b_inferred["j"] = b_inferred["c1"] + 52 * 3
        b_inferred["a"] = b_inferred["c1"] + 52 * 10
    dates = list(a["date"])
    status = list(a["age_class"])
    for j in range(len(dates)): # Treat all calves 36+ months old as juveniles.
        dates[j] = to_iso(dates[j])
        if status[j] == "[36+ mo calf]":
            status[j] = "[24-36 mo]"
            
            
    current_state = status[0] # The first status that a dolphin is spotted at.
    if current_state == "[10+ yr adult]":
        age_dict[i]["a"] = time_diff(lower_bound, dates[0], granularity)
        continue # If we spot an adult, record the time spotted, and we're done. Ignore deaths for now.
    elif current_state == "[juvenile]":
        # If we spot a juvenile, record the time spotted, and see if we can find an adult status.
        age_dict[i]["j"] = time_diff(lower_bound, dates[0], granularity)
        adult_age = find_first_time(dates, status, "[10+ yr adult]")
        if adult_age != "NA": # If an adult status is found, just list it.
            age_dict[i]["a"] = time_diff(lower_bound, adult_age, granularity)
        else:
            age_dict[i]["a"] = age_dict[i]["j"] + 7*52 # Otherwise, just project a faraway age.
    elif current_state == "[24-36 mo]":
        # For C3 calves, they can age up into juveniles, but do not have the time to become adults.
        age_dict[i]["c3"] = time_diff(lower_bound,dates[0], granularity)
        j_age = find_first_time(dates, status, "[juvenile]")
        if j_age != "NA":
            age_dict[i]["j"] = time_diff(lower_bound, j_age, granularity)
            if age_dict[i]["j"] > age_dict[i]["c3"] + 52:
                age_dict[i]["j"] = age_dict[i]["c3"] + 52
        else:
            age_dict[i]["j"] = age_dict[i]["c3"] + 52
    elif current_state == "[12-24 mo]":
        # For C2 calves, they can age up into juveniles, and also may age up into c3.
        age_dict[i]["c2"] = time_diff(lower_bound, dates[0], granularity)
        j_age = find_first_time(dates, status, "[juvenile]")
        c3_age = find_first_time(dates, status, "[24-36 mo]")
        if c3_age != "NA":
            age_dict[i]["c3"] = time_diff(lower_bound, c3_age, granularity)
            if age_dict[i]["c3"] > age_dict[i]["c2"] + 52:
                age_dict[i]["c3"] = age_dict[i]["c2"] + 52
        else:
            age_dict[i]["c3"] = age_dict[i]["c2"] + 52
        if j_age != "NA":
            age_dict[i]["j"] = time_diff(lower_bound, j_age, granularity)
            if age_dict[i]["j"] > age_dict[i]["c3"] + 52:
                age_dict[i]["j"] = age_dict[i]["c3"] + 52
        else:
            age_dict[i]["j"] = age_dict[i]["c3"] + 52
    elif current_state == "[0-12 mo]":
        # For C2 calves, they can age up into juveniles, and also may age up into 
        age_dict[i]["c1"] = time_diff(lower_bound, dates[0], granularity)
        j_age = find_first_time(dates, status, "[juvenile]")
        c3_age = find_first_time(dates, status, "[24-36 mo]")
        c2_age = find_first_time(dates, status, "[12-24 mo]")
        if c2_age != "NA":
            age_dict[i]["c2"] = time_diff(lower_bound, c2_age, granularity)
            if age_dict[i]["c2"] > age_dict[i]["c1"] + 52:
                age_dict[i]["c2"] = age_dict[i]["c1"] + 52
        else:
            age_dict[i]["c2"] = age_dict[i]["c1"] + 52
        if c3_age != "NA":
            age_dict[i]["c3"] = time_diff(lower_bound, c3_age, granularity)
            if age_dict[i]["c3"] > age_dict[i]["c2"] + 52:
                age_dict[i]["c3"] = age_dict[i]["c2"] + 52
        else:
            age_dict[i]["c3"] = age_dict[i]["c2"] + 52
        if j_age != "NA":
            age_dict[i]["j"] = time_diff(lower_bound, j_age, granularity)
            if age_dict[i]["j"] > age_dict[i]["c3"] + 52:
                age_dict[i]["j"] = age_dict[i]["c3"] + 52
        else:
            age_dict[i]["j"] = age_dict[i]["c3"] + 52
    else:
        print("ERROR: Unknown status: " + str(current_state))

# Finally, let's figure out everyone's demo_info.
# In this setting, just consider those who are grown and those who are 
demo_info = {i:{j: "NA" for j in ["birth", "juv", "grown"]} for i in range(len(node_keys))}
for i in range(len(node_keys)):
    # First, check if c1 exists.
    if age_dict[i]["c1"] > 0:
        demo_info[i]["birth"] = age_dict[i]["c1"]
    else: # If we don't HAVE a birthday, figure out if we see a c2 or a c3 close enough.
        if age_dict[i]["c2"] > 2 * 52:    
            demo_info[i]["birth"] = age_dict[i]["c2"] - 2 * 52
        elif age_dict[i]["c3"] > 3 * 52:
            demo_info[i]["birth"] = age_dict[i]["c3"] - 3 * 52
        else:
            demo_info[i]["birth"] = -1 # Otherwise, we were born before the study started.
    # Next, deal with the transitionary period.
    # Note that we do NOT deal with death here. 
    

    if age_dict[i]["j"] >= 0:
        if (age_dict[i]["c1"] != -1000 or age_dict[i]["c2"] != -1000 or age_dict[i]["c3"] != -1000): 
        # If we were ever seen as a calf, directly use the observed transition
            demo_info[i]["juv"] = age_dict[i]["j"]
        # If we were NEVER seen as a calf, then say we transitioned before.
        else:
            demo_info[i]["juv"] = -1
    elif age_dict[i]["j"] > -1000: # If we saw an earlier truncation period, keep it at -1.
        demo_info[i]["juv"] = -1
    elif age_dict[i]["a"] != -1000: # If we never saw it at juvenile, but only saw it as an adult...
        demo_info[i]["juv"] = -1
    elif age_dict[i]["c3"] != -1000: # If we saw a c3, but not an adult or juvenile sighting, just add a year to our c3.
        demo_info[i]["juv"] = age_dict[i]["c3"] + 52
    elif age_dict[i]["c2"] != -1000: # If we only saw c2 and not c3, then we add two years to get our transition.
        demo_info[i]["juv"] = age_dict[i]["c2"] + 52 * 2
        #print("Dolphin " + str(i) + " may have died")
    elif age_dict[i]["c1"] != -1000: # If we saw only c1, then add three years.
        demo_info[i]["juv"] = age_dict[i]["c1"] + 52 * 3
        #print("Dolphin " + str(i) + " may have died")
        
    if age_dict[i]["a"] >= 0:
        if (age_dict[i]["j"] != -1000): 
        # If we were ever seen as a juvenile, directly use the observed transition
            demo_info[i]["grown"] = age_dict[i]["a"]
        # If we were NEVER seen as a juvenile, then say we transitioned before.
        else:
            demo_info[i]["grown"] = -1
    elif age_dict[i]["a"] > -1000: # If we saw an earlier truncation period, keep it at -1.
        demo_info[i]["grown"] = -1
    elif age_dict[i]["a"] != -1000: # If we never saw it at juvenile, but only saw it as an adult...
        demo_info[i]["grown"] = -1
    elif age_dict[i]["c3"] != -1000: # If we saw a c3, but not an adult or juvenile sighting, just add a year to our c3.
        demo_info[i]["grown"] = age_dict[i]["c3"] + 52 * 8
    elif age_dict[i]["c2"] != -1000: # If we only saw c2 and not c3, then we add nine years to get our transition.
        demo_info[i]["grown"] = age_dict[i]["c2"] + 52 * 9
        #print("Dolphin " + str(i) + " may have died")
    elif age_dict[i]["c1"] != -1000: # If we saw only c1, then add ten years.
        demo_info[i]["grown"] = age_dict[i]["c1"] + 52 * 10
        #print("Dolphin " + str(i) + " may have died")

# Analyze the infection origins.
origin_dates = []
origin_deltas = []
for d in infection_origin:
    a = orig_data[orig_data["dolphin_id"] == d]
    dates = list(a["date"])
    status = list(a["infection_status"])
    pre_sighting = ("NA", "NA")
    for i in range(len(dates)):
        if status[i] == 1:
            origin_dates.append(dates[i])
            delta_days = datetime.date.fromisoformat(dates[i]) - datetime.date.fromisoformat(lower_bound)
            week = int(delta_days.days)//int(granularity)
            origin_deltas.append(week)
            break        



final_pickle = {"i_list": index_infection_origin, "r_list": index_recovered_origin, "candidates": candidates, "demo_info": demo_info, "i_times": origin_deltas, "network_transition_times": network_transition_times, "time_steps": total_slots, "test_times": test_times_dict, "results": results_dict, "X": X, "X_dynamic": X_dynamic, "node_keys": node_keys, "first_date": lower_bound, "last_date": upper_bound, "granularity": granularity}
dump_file = open("C:/dolphin_analysis/dolphin_data.pkl", "wb")
pickle.dump(final_pickle, dump_file)
dump_file.close()


