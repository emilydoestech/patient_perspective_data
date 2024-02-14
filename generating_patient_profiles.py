## Generating Patient Profiles
"""
We want to generate patient profiles which are as realistic as possible for
  both the game scenarios and for the patient matching algorithm. In order
  to do this, we separate the data into the four categories:
  - **specific clinical findings** (traits that are not observable by the
    patient and would _only_ be identified through _specific_ investigations):
      - **require specific symptoms** relating to the clinical finding
  - **(non-specific) clinical findings** (traits that are not observable by
    the patient which would be identified through _routine_ investigations ):
    - **do not require specific symptoms**
  - **symptoms** (traits observable by the patient)
  - **developmental traits** (traits noticable from birth or a young age)

 We generate the traits in each of the four categories listed above based on
  the frequency of these traits occuring in a patient with a given disorder.

 First, we generate clinical findings.

 For clinical findings that would only usually be identified through specific
  investigations, we find symptoms that would prompt these investigations.

 We do this by identifying the clinical findings which require prior symptoms
 and generating symptoms which relate to the clinical finding (we pick 1-2
   symptoms relating to the finding)

 We then generate the remaining symptoms and developmental traits based on
  their frequency within a given disorder.

 The list is then reordered so that participants will initially see
  developmental traits, then symptoms, then clinical findings.
"""

# import required packages
import pandas as pd
import numpy as np
import random
import names
from math import fsum
from itertools import permutations


df = pd.read_csv('patient_perspective_data.csv')
# remove phenotypes that do not occur
df = df[df['probability'] != 0]
# remove redundent phenotypes(e.g. duplicate phenotypes)
df = df[df['discovery_group'] != 'R']



"""
## Normalising The Data for a Given Category

In order to use np.random.choice, we need to normalise the phenotype frequency.
So, let us define a function which normalises the data for each of the four
 categories of phenotypes for a given disorder.
This enables us to generate the phenotypes separately to control the order.
We do this by taking a column name and desired value as inputs so that we pick a
 subset of all data where `column=specified_val` based on the frequency of these
 symptoms occuring within said disorder.
"""

def phenotype_choice(df,
                     disorder_name: str,
                     column_name:str,
                     column_value:str,
                     total_phenotypes:int):
    # Reduce data to inputted disorder and reset index to prevent indexing
    #  errors
    current_disorder = df[df.disorder == disorder_name].reset_index(drop=True)
    # Reducing data to a specific attribute so that a column = attribute
    #  (e.g. if column=Label columnVal=F would reduce data to clinical
    #  findings)
    current_discovery_grp = current_disorder[current_disorder[column_name]
                                          == column_value]
    # We take the list of phenotypes
    a = current_discovery_grp.patient_name
    # We take the probability of each phenotype
    p = current_discovery_grp.probability
    # We change the number of phenotypes sampled (size) to be representative
    #  of the weight of the subset of phenotypes
    subset_weight = fsum(p)/fsum(current_disorder.probability)
    size = total_phenotypes*subset_weight
    # Since the number of each type of phenotype will vary from patient to
    #  patient, we add noise to change how many phenotypes are generated and
    #  then round to an integer
    size = round(np.random.normal(size, size*0.15))
    # An empty numpy array is identity under concatination to allow for
    # disorders with no phenotypes of a given disorder group
    if len(a) == 0:
        phenotypes = np.empty((0,))
    else:
        if len(a)<size:
            size = round(len(a)*0.8)
        # sample phenotypes based on normalised probability
        phenotypes = np.random.choice(a,
                                      size=size,
                                      replace=False,
                                      p=p/fsum(p))
    return(phenotypes)

"""
## Generating Patient Profiles for a Single Patient

Here we use phenotypeChoice to generate clinical findings, symptoms and
 developmental traits.

For clinical findings that would only usually be identified through specific
 investigations, we find symptoms that would prompt these investigations.
We do this by identifying the clinical findings which require prior symptoms
 and generating symptoms which relate to the clinical finding (we pick 1-2
 symptoms relating to the finding - may need to change this).
"""

def generate_single_patient(disorder: str, total_phenotypes:int, patient_info):
    # Generating findings, symptoms and developmental traits
    fin = phenotype_choice(df,
                           disorder,
                           'discovery_group',
                           'F',
                           total_phenotypes)
    sym = phenotype_choice(df,
                           disorder,
                           'discovery_group',
                           'S',
                           total_phenotypes)
    dev = phenotype_choice(df,
                           disorder,
                           'discovery_group',
                           'D',
                           total_phenotypes)
    # Find data for all clinical findings that would only usually be
    #  identified through specific investigations so that we can generate
    #  symptoms/findings that would promt these investigations
    current_disorder = df[df.disorder == disorder].reset_index(drop=True)
    fin_array = current_disorder[current_disorder.patient_name.isin(fin)]
    specific_fin = fin_array[fin_array.prerequisite_needed == 'Y']
    # Each of these findings have pre-requisites listed in the
    specific_fin = specific_fin.prerequisite_type
    # An empty dataframe is identity under concatination, so we can
    # concatinate the dataframe for each finding with pre-requisites
    sampled_prereq_list = np.empty((0,))
    if specific_fin.shape != (0,):
        for finding in specific_fin:
            # we select phenotypes which relate to the findings
            prereq_df = df[df.prerequisite_needed == 'N']
            prereq_finding = phenotype_choice(prereq_df,
                                             disorder,
                                             'HPO_category',
                                             finding,
                                             total_phenotypes)
            # We concatinate each list of pre-requisites (for different
            #  specific findings) into one list if it is not in the current
            #  list of symptoms or developmental traits
            for sampled_prereq in prereq_finding:
                if np.isin(np.hstack((sym,dev)),
                           sampled_prereq).any == False:
                    sampled_prereq_list = np.hstack((sampled_prereq_list,
                                                     sampled_prereq))
        if sym.shape != (0,):
            sym = np.hstack((sym, sampled_prereq_list))
        else:
            sym = sampled_prereq_list
    # We concatinate all phenotypes ordering them to how they should be
    #  displayed to the user, since developmental traits would naturally
    #  occur first followed by symptoms, followed by clinical findings
    phenotypes = np.hstack((dev, sym, fin))
    # Concatinate patient information for lab study or for peer matching

    phenotypes = np.append(patient_info, phenotypes)
    return(phenotypes)


## Generating Time-Series Personas for Lab Study

def generate_timeseries_personas(disorder_list: list,
                                  total_phenotypes:int,
                                  total_participants:int):
    # add noise to vary the number of phenotypes sampled per user
    total_phenotypes = round(np.random.normal(total_phenotypes,
                                              total_phenotypes*0.1))
    # An empty dataframe is identity under concatination, so we can
    #  concatinate the dataframe for each game
    games = pd.DataFrame()
    # Generate games for a given number of participants
    for i in range(0,total_participants):
        j=(i+1)%5
        website_choices = list(permutations([0,1,2]))[j]
        # 0: peer matching, 1: maladyHelp, 2: Google custom
        # Each participant should play the game three times for each
        #  disorder, so we number the games using n
        n=0
        for disorder in disorder_list:
            # Here we create a dataframe showing participant ID, game ID,
            #  and a random name for their patient case
            patient_info = pd.DataFrame()
            single_game = generate_single_patient(disorder,
                                                  total_phenotypes,
                                                  patient_info)
            round1, round2, round3 = np.array_split(single_game, 3)
            patient_info = pd.DataFrame([i, n, disorder, website_choices[n]])
            n+=1
            # Join the array of phenotypes into a single string of
            #  phenotypes for each round, separated with a comma
            round1 = ", ".join(round1)
            round2 = ", ".join(round2)
            round3 = ", ".join(round3)
            rounds = pd.DataFrame([round1,round2,round3])
            game = pd.concat([patient_info, rounds])
            games = pd.concat([games, game], axis=1)
    # we transpose the dataframe for readability
    games = games.T.reset_index(drop=True)
    games.columns=['participant_ID', 'game_ID', 'disorder',
                   'website_choice', 'round_1', 'round_2', 'round_3']
    return(games)



## Generating Users for Peer Matching Algorithm

def generate_users(disorder_list: list,
                   total_phenotypes:int,
                   users_per_disorder:int):
    # An empty dataframe is identity under concatination, so we can
    #  concatinate the dataframe for each user
    users = pd.DataFrame()
    # generate a given number of disorders
    for i in range(0,users_per_disorder):
        # currently generates an equal number of users per disorder
        for disorder in disorder_list:
            # We want the majority of users (currently set to 80%) in the
            #  matching algorithm to be undiagnosed
            k = random.randint(1, 10)
            if k>2:
                # We assign users names and diagnosis status
                patient_info = pd.DataFrame([names.get_first_name(),
                                             'undiagnosed'])
            else:
                # We assign users names and diagnosis
                patient_info = pd.DataFrame([names.get_first_name(),
                                             disorder])
            # Generate single user
            user = generate_single_patient(disorder,
                                           total_phenotypes,
                                           patient_info)
            # We remove the end phenotypes for users at an earlier stage
            #  of diagnosis
            phenotypes = len(user)
            if k>7:
                user = user[0:round(phenotypes*0.75)]
            elif k>4:
                user = user[0:round(phenotypes*0.85)]
            user = pd.DataFrame(user)
            # Concatinate to one dataframe
            users = pd.concat([users, user], axis=1)
    # we transpose the dataframe for readability
    users = users.T.reset_index(drop=True)
    # removing empty columns resulting from our removal of phenotypes
    users = users.dropna(how='all')
    return(users)


# Generate the Time Series Persona Dataset (for lab study)
study_disorders = ['Hypermobile Ehlers-Danlos syndrome',
                   'Fabry disease',
                   'Gaucher disease']
time_series_personas = generate_timeseries_personas(study_disorders, 10, 100)

# Generate the Static User Profile Dataset (for peer matching)

# We generate 25 users for each of the 19 disorders in our database
# with an average of 10 phenotypes
peer_matching_disorders = df.disorder.unique()
static_user_base = generate_users(peer_matching_disorders, 10, 25)


# In[ ]:
