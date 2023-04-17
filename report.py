from datetime import datetime
import pathlib
import glob
import os.path
import shutil

#lists for storing the activity of every event
sysmon_list = []
track_list = []
comm_list = []


#************************************COMMUNICATION MODULE************************************
def getCommScore(comm_list):
    #first split on TARGET
    split_list = []
    response_times = []
    repsonse_accuracies = []
    i=0
    while i < len(comm_list):
        if comm_list[i][6] == 'TARGET':
            j = i+1
            while j < len(comm_list) and comm_list[j][6] != 'TARGET':
                j+=1
            split_list.append(comm_list[i:j])
            i = j
            continue
        i+=1

    #traverse through the split list and get the score
    for event in split_list:
        channel =''
        #case when the prompt is for the subject
        if event[0][4] == 'OWN':
            channel = event[0][5]
            curr_target = event[0][7].strip()
            #when ENTER is not pressed
            if event[-1][5] != 'RETURN':
                #add 30 seconds to the response time
                #add 0 to the accuracy
                response_times.append(30)
                repsonse_accuracies.append(0)
                continue

            #when ENTER is pressed
            #get the last set frequency
            last_freq = event[-2][6].strip()
            if(curr_target == last_freq):
                #correct frequency is set
                repsonse_accuracies.append(1)
                start_time = datetime.strptime(event[0][0], '%H:%M:%S.%f')
                end_time = datetime.strptime(event[-2][0], '%H:%M:%S.%f')
                response_time = end_time - start_time
                response_times.append(response_time.total_seconds())
                continue

            #correct frequency is not set
            error = abs(float(curr_target) - float(last_freq))/float(curr_target)
            accuracy = 1 - error
            repsonse_accuracies.append(accuracy)
            start_time = datetime.strptime(event[0][0], '%H:%M:%S.%f')
            end_time = datetime.strptime(event[-2][0], '%H:%M:%S.%f')
            response_time = end_time - start_time
            response_times.append(response_time.total_seconds())

        #case when the prompt is not for the subject
        if event[0][4] == 'OTHER':
            channel = event[0][5]
            #check if the frequency is changed
            if(len(event)>=5):
                #frequency is changed
                response_times.append(30)
                repsonse_accuracies.append(0)

            else:
                #frequency is not changed
                response_times.append(0)
                repsonse_accuracies.append(1)

    print(f'response times:{response_times}')
    print(f'response accuracies:{repsonse_accuracies}')
#*************************************************************************************


#************************************SYSMON MODULE************************************
def getSysmonScore(sysmon_list):
    total_failures = 0
    total_hits = 0
    false_alarm = 0
    response_time = 0

    for i,event in enumerate(sysmon_list)   :
        if event[5] == 'FAILURE\n':
            total_failures += 1
            #check if the next event is a hit
            if(sysmon_list[i+1][5]=='HIT\n'):
                total_hits += 1

            #check if the next event is a miss
            if(sysmon_list[i+1][5]=='MISS\n'):
                false_alarm += 1

            #calculate response time
            time1 = datetime.strptime(sysmon_list[i][0], '%H:%M:%S.%f')
            time2 = datetime.strptime(sysmon_list[i+1][0], '%H:%M:%S.%f')
            res = time2 - time1
            print(res.total_seconds())
            response_time += res.total_seconds()

    avg_response_time = response_time/total_failures
    print('Total Failures: ',total_failures)
    print('Total Hits: ',total_hits)
    print('False Alarms: ',false_alarm)
    print('Average Response Time: ',avg_response_time)

#*************************************************************************************


#************************************TRACKING MODULE**********************************
def getTrackScore(track_list):

    split_list = []
    count = 0
    i=0
    total_time_out = []
    mean_deviation_list = []

    while i <len(track_list):
        if track_list[i] != 'AUTO\n':
            j = i
            temp_list=[]
            while(j<len(track_list) and track_list[j]!='AUTO\n'):
                temp_list.append(track_list[j])
                j+=1
            split_list.append(temp_list)
            i = j
        i+=1

    for event in split_list:
        # print(type(event[0]))
        temp_mean =0.0
        time_in = 0
        time_out = 0
        for i in range(len(event)):
            event_dict = eval(event[i])
            if(i==0):
                time_in = event_dict['total']['time_out_ms']
            if(i==len(event)-1):
                time_out = event_dict['total']['time_out_ms']
            temp_mean += event_dict['total']['deviation_mean']

        mean_deviation_list.append(temp_mean/len(event))
        total_time_out.append(time_out-time_in)
        # print(time_out-time_in)

    print('Mean Deviation: ',mean_deviation_list)
    print('Total Time Out: ',total_time_out)
    # [print(ele) for ele in split_list]

#*************************************************************************************

if __name__ == '__main__':
    participant_info = []

    #get current date and time
    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y_%H-%M")

    #get user ID and session from the sync file
    file = open('sync.txt','r')
    for line in file:
        line = line.strip().split('-')
        participant_ID = line[0].split(':')[1]
        session = 'session_'+str(line[1])

    #create folder for the participant in User_data
    data_path = pathlib.Path('User_data')
    participant_path = data_path/participant_ID
    participant_path.mkdir(parents=True, exist_ok=True)

    #create the folder for the current session
    session_path = participant_path/session
    session_path.mkdir(parents=True, exist_ok=True)

    #save the latest log file, pulse_data.txt, Open BCI data in the session folder

    #get the latest log file
    folder_path = pathlib.Path('Logs')
    file_type = r'*.log'
    files = glob.glob(str(folder_path/file_type))
    log_file_path = max(files, key=os.path.getctime)

    #save the log file in the session folder
    prefix_file_name = participant_ID+'_'+session+'_'+date_string
    # prefix_file_name = participant_ID+'_'+session
    new_log_file = prefix_file_name+'.log'
    shutil.copy(log_file_path,session_path/new_log_file)

    #get the latest OpenBCI folfer
    openbci_folder_path  = pathlib.Path('C:/Users/Abhishek/Documents/OpenBCI_GUI/Recordings')
    openbci_file_type = r'*'
    openbci_files = glob.glob(str(openbci_folder_path/openbci_file_type))
    openbci_current_folder = max(openbci_files, key=os.path.getctime)

    #get the latest OpenBCI file
    openbci_current_folder = pathlib.Path(openbci_current_folder)
    openbci_file_type = r'*.txt'
    openbci_files = glob.glob(str(openbci_current_folder/openbci_file_type))
    openbci_file_path = max(openbci_files, key=os.path.getctime)

    #save the OpenBCI file in the session folder
    new_openbci_file = prefix_file_name+'.txt'
    shutil.copy(openbci_file_path,session_path/new_openbci_file)

    #move the pulse data to the session folder
    pulse_data_path = 'pulse_data.txt'
    new_pulse_file = prefix_file_name+'_pulse_data.txt'
    shutil.copy(pulse_data_path,session_path/new_pulse_file)

    #move the track data to the session folder
    track_data_path = 'track_log.txt'
    new_track_file = prefix_file_name+'_track_log.txt'
    shutil.copy(track_data_path,session_path/new_track_file)


    # traverse thorugh the log file for sysmon and communication events
    # with open(log_file_path ,'r') as f:
    #     for line in f:
    #         line = line.split('\t')

    #         if(len(line)<=1):
    #             continue

    #         if line[2] == 'SYSMON':
    #             sysmon_list.append(line)

    #         if line[2] == 'COMMUN' and line[6]!='SELECTED\n':
    #             comm_list.append(line)

    # # traverse thorugh the log file for tracking events
    # track_log_file_path = 'track_log.txt'
    # with open(track_log_file_path ,'r') as tf:
    #     for line in tf:
    #         track_list.append(line)

# [print (i) for i in comm_list]
# print("****************************************")
# print("TRACK MODULE SCORES")
# getTrackScore(track_list)
# print("****************************************")
# print("COMM MODULE SCORES")
# getCommScore(comm_list)
# print("****************************************")
# print("SYSMON MODULE SCORES")
# getSysmonScore(sysmon_list)
# print("****************************************")
