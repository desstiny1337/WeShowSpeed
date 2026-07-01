
import socket
import sys
import getopt
import os
import time
import csv
import math
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

PI= 3.14159265359

data_size = 2**17

ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return '' # No width!
    if x<mn: x= mn      # Clip to bounds.
    if x>mx: x= mx      # Clip to bounds.
    tx= mx-mn # Total real units possible to show on graph.
    if tx<=0: return 'backwards' # Stupid bounds.
    upw= tx/float(w) # X Units per output char width.
    if upw<=0: return 'what?' # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0: # Then there is a negative part to graph.
        if x < 0: # And the plot is on the negative side.
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else: # Plot is on pos. Neg side is empty.
            negnonpu= -mn + min(0,mx) # But still show some empty neg.
    if mx > 0: # There is a positive part to the graph
        if x > 0: # And the plot is on the positive side.
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else: # Plot is on neg. Pos side is empty.
            posnonpu= mx - max(0,mn) # But still show some empty pos.
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False):
        self.vision = vision

        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 # "Maximum number of learning episodes to perform"
        self.trackname= 'unknown'
        self.stage= 3 # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug= False
        self.maxSteps= 100000  # 50steps/second
        self.parse_the_command_line()
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d
        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg='%s(init %s)' % (self.sid,a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('torcs -nofuel -nodamage -nolaptime &')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')

                    time.sleep(1.0)
                    os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                       ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'debug','help','version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug= True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host= opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid= opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname= opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage= int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port= int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes= int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps= int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                                       opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        if not self.so: return
        sockdata= str()

        while True:
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. "+
                        "You were in %d place.") %
                        (self.port,self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata: # Empty?
                continue       # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H") # Clear for steady output.
                    print(self.S)
                break # Can now return from this function.

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1],str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps,self.port)))
        self.so.close()
        self.so = None

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

    def __repr__(self):
        return self.fancyout()
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out= str()
        sensors= [ # Select the ones you want in the order you want them.
        'stucktimer',
        'fuel',
        'distRaced',
        'distFromStart',
        'opponents',
        'wheelSpinVel',
        'z',
        'speedZ',
        'speedY',
        'speedX',
        'targetSpeed',
        'rpm',
        'skid',
        'slip',
        'track',
        'trackPos',
        'angle',
        ]

        for k in sensors:
            if type(self.d.get(k)) is list: # Handle list type data.
                if k == 'track': # Nice display for track sensors.
                    strout= str()
                    raw_tsens= ['%.1f'%x for x in self.d['track']]
                    strout+= ' '.join(raw_tsens[:9])+'_'+raw_tsens[9]+'_'+' '.join(raw_tsens[10:])
                elif k == 'opponents': # Nice display for opponent sensors.
                    strout= str()
                    for osensor in self.d['opponents']:
                        if   osensor >190: oc= '_'
                        elif osensor > 90: oc= '.'
                        elif osensor > 39: oc= chr(int(osensor/2)+97-19)
                        elif osensor > 13: oc= chr(int(osensor)+65-13)
                        elif osensor >  3: oc= chr(int(osensor)+48-3)
                        else: oc= '?'
                        strout+= oc
                    strout= ' -> '+strout[:18] + ' ' + strout[18:]+' <-'
                else:
                    strlist= [str(i) for i in self.d[k]]
                    strout= ', '.join(strlist)
            else: # Not a list type of value.
                if k == 'gear': # This is redundant now since it's part of RPM.
                    gs= '_._._._._._._._._'
                    p= int(self.d['gear']) * 2 + 2  # Position
                    l= '%d'%self.d['gear'] # Label
                    if l=='-1': l= 'R'
                    if l=='0':  l= 'N'
                    strout= gs[:p]+ '(%s)'%l + gs[p+3:]
                elif k == 'damage':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,10000,50,'~'))
                elif k == 'fuel':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,100,50,'f'))
                elif k == 'speedX':
                    cx= 'X'
                    if self.d[k]<0: cx= 'R'
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-30,300,50,cx))
                elif k == 'speedY': # This gets reversed for display to make sense.
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k]*-1,-25,25,50,'Y'))
                elif k == 'speedZ':
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-13,13,50,'Z'))
                elif k == 'z':
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k],.3,.5,50,'z'))
                elif k == 'trackPos': # This gets reversed for display to make sense.
                    cx='<'
                    if self.d[k]<0: cx= '>'
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k]*-1,-1,1,50,cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout= '%3d %s' % (self.d[k], bargraph(self.d[k],0,300,50,"'"))
                    else: strout= 'Not stuck!'
                elif k == 'rpm':
                    g= self.d['gear']
                    if g < 0:
                        g= 'R'
                    else:
                        g= '%1d'% g
                    strout= bargraph(self.d[k],0,10000,50,g)
                elif k == 'angle':
                    asyms= [
                          "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                          "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                          "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                          "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."  ]
                    rad= self.d[k]
                    deg= int(rad*180/PI)
                    symno= int(.5+ (rad+PI) / (PI/12) )
                    symno= symno % (len(asyms)-1)
                    strout= '%5.2f %3d (%s)' % (rad,deg,asyms[symno])
                elif k == 'skid': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    skid= 0
                    if frontwheelradpersec:
                        skid= .5555555555*self.d['speedX']/frontwheelradpersec - .66124
                    strout= bargraph(skid,-.05,.4,50,'*')
                elif k == 'slip': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    slip= 0
                    if frontwheelradpersec:
                        slip= ((self.d['wheelSpinVel'][2]+self.d['wheelSpinVel'][3]) -
                              (self.d['wheelSpinVel'][0]+self.d['wheelSpinVel'][1]))
                    strout= bargraph(slip,-5,150,50,'@')
                else:
                    strout= str(self.d[k])
            out+= "%s: %s\n" % (k,strout)
        return out

class DriverAction():
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''
    def __init__(self):
       self.actionstr= str()
       self.d= { 'accel':0.2,
                   'brake':0,
                  'clutch':0,
                    'gear':1,
                   'steer':0,
                   'focus':[-90,-45,0,45,90],
                    'meta':0
                    }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)
        self.d['clutch']= clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear']= 0
        if self.d['meta'] not in [0,1]:
            self.d['meta']= 0
        if type(self.d['focus']) is not list or min(self.d['focus'])<-180 or max(self.d['focus'])>180:
            self.d['focus']= 0

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            if not type(v) is list:
                out+= '%.3f' % v
            else:
                out+= ' '.join([str(x) for x in v])
            out+= ')'
        return out
        return out+'\n'

    def fancyout(self):
        '''Specialty output for useful monitoring of bot's effectors.'''
        out= str()
        od= self.d.copy()
        od.pop('gear','') # Not interesting.
        od.pop('meta','') # Not interesting.
        od.pop('focus','') # Not interesting. Yet.
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout=''
                strout= '%6.3f %s' % (od[k], bargraph(od[k],0,1,50,k[0].upper()))
            elif k == 'steer': # Reverse the graph to make sense.
                strout= '%6.3f %s' % (od[k], bargraph(od[k]*-1,-1,1,50,'S'))
            else:
                strout= str(od[k])
            out+= "%s: %s\n" % (k,strout)
        return out

def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]

def drive_example(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S,R= c.S.d,c.R.d
    target_speed=160

    R['steer']= S['angle']*25 / PI
    R['steer']-= S['trackPos']*.25

    R['accel'] = max(0.0, min(1.0, R['accel']))
    

    if S['speedX'] < target_speed - (R['steer']*2.5):
        R['accel']+= .4
    else:
        R['accel']-= .2
    if S['speedX']<10:
       R['accel']+= 1/(S['speedX']+.1)

    if ((S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0]+S['wheelSpinVel'][1]) > 2):
       R['accel']-= 0.1



    R['gear']=1
    if S['speedX']>60:
        R['gear']=2
    if S['speedX']>100:
        R['gear']=3
    if S['speedX']>140:
        R['gear']=4
    if S['speedX']>190:
        R['gear']=5
    if S['speedX']>220:
        R['gear']=6
    return

# if __name__ == "__main__":
#     C= Client(p=3001)
#     for step in range(C.maxSteps,0,-1):
#         C.get_servers_input()
#         drive_example(C)
#         C.respond_to_server()
#     C.shutdown()

# MODULAR TORCS DRIVER
# Ordered layout:
# 1. User configuration
# 2. Runtime globals
# 3. Small utilities and state update
# 4. Neural model functions
# 5. Logging / audit functions
# 6. Rule-based control functions
# 7. Main drive function
# 8. Main loop
# =============================================================================


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# -----------------------------
# Rule-based driver parameters
# -----------------------------
TARGET_SPEED = 290
STEER_GAIN = 22
CENTERING_GAIN = 0.24
BRAKE_THRESHOLD = 0.38
GEAR_SPEEDS = [0, 45, 95, 135, 175, 235]
ENABLE_TRACTION_CONTROL = True


# -----------------------------
# Main mode switches
# -----------------------------
# Safe default:
#   True  -> pure stable rule-based driver, no neural calls.
#   False -> neural debug / safe neural correction may run depending on switches below.
ENABLE_RULE_BASED_ONLY = True

# Dataset logging for training / analysis.
ENABLE_LOGGING = False
LOG_FILE = "baseline_1_51_85_clean_20laps.csv"

ENABLE_BC_STATE_LOG = False
BC_STATE_LOG_FILE = "bc_state_for_offline_compare_15185.csv"
BC_STATE_LOG_EVERY_N_STEPS = 1

# Control black-box audit.
# Useful when checking why the car starts wobbling or leaves track.
ENABLE_CONTROL_AUDIT = False
CONTROL_AUDIT_FILE = "control_audit.csv"

# -----------------------------
# Fast-path / 1:21 engineering infrastructure
ENABLE_SECTOR_TRACE = True
SECTOR_TRACE_FILE = "sector_trace.csv"
SECTOR_SIZE_M = 25.0
TRACK_LENGTH_M = 3608.0

ENABLE_DYNAMIC_TARGET_SPEED = False
DYNAMIC_TARGET_BLEND = 0.20

# Keep False until we build sector_profile.json from real logs.
ENABLE_SECTOR_SPEED_PROFILE = True
SECTOR_PROFILE_FILE = "sector_profile_v15_exit_plus_aggressive.json"
SECTOR_SPEED_PROFILE = None

ENABLE_TRACK_PLAN = True
TRACK_PLAN_FILE = "track_plan_active.json"
TRACK_PLAN = None
TRACK_PLAN_BLEND = 1.06
PREVIEW_BRAKE_ENABLED = False
PREVIEW_LOOKAHEAD_M = (75.0, 125.0, 175.0, 225.0)

V23_UPSHIFT_RPM = 9000.0
V23_DOWNSHIFT_RPM = 5400.0
V23_APEX_STEER_GAIN = 0.0180
V23_EXIT_UNWIND = 0.1800
V23_EXIT_MIN_ACCEL = 1.000
V23_STRAIGHT_MIN_ACCEL = 1.000
V23_EXIT_MAX_ANGLE = 0.260


V28_RPM_VIRTUAL_SCALE = 1.000
V28_UPSHIFT_VIRTUAL_RPM = 18000.0
V28_DOWNSHIFT_VIRTUAL_RPM = 12000.0
V28_SHIFT_LOCK_STEPS = 6
V28_ACCEL_PRESS_STEP = 1.000
V28_EXIT_MAX_ANGLE = 0.340
V28_STRAIGHT_FRONT_MIN = 82.0
V28_EXIT_FRONT_MIN = 34.0
V28_LAST_GEAR_CMD = 1
V28_SHIFT_LOCK = 0


V29_EARLY_BRAKE_STRENGTH = 1.120
V29_EARLY_BRAKE_ENABLED = True

V34_SECTOR77_GUARD_ENABLED = True
V34_SECTOR77_BRAKE_STRENGTH = 1.250
V34_EDGE_BRAKE_HIGH = 0.480
V34_EDGE_BRAKE_MID = 0.360
V34_PRE_FLOOR_1 = 0.090
V34_PRE_FLOOR_2 = 0.200
V34_PRE_FLOOR_3 = 0.380

V35_CENTERED_CORNER_ENABLED = True
V35_BRAKE_STRENGTH = 1.200
V35_TURNIN_GAIN = 0.0500
V35_MIDCORNER_MIN_ACCEL = 0.820
V35_EXIT_MIN_ACCEL = 1.000
V35_BRAKE_CAP = 0.740


V36_PROFILE_NAME = "balanced"
FINAL_TURN_GUARD_ENABLED = True
FINAL_TURN_MAX_BLEND = 0.56
FINAL_TURN_START_M = 3050.0
FINAL_TURN_END_M = 3325.0
FINAL_TURN_STEER_ASSIST_ENABLED = True

# Console spam is dangerous during TORCS runs, keep False by default.
PRINT_DEBUG = False


# -----------------------------
# Neural configuration
# -----------------------------
# Recommended online workflow:
#   1) Keep ENABLE_RULE_BASED_ONLY = True until rule driver is stable.
#   2) Set ENABLE_RULE_BASED_ONLY = False and USE_BC_DEBUG_ONLY = True
#      to measure neural inference lag WITHOUT applying neural controls.
#   3) Only then set USE_SAFE_STEER_CORRECTION = True for tiny steer correction.
USE_NEURAL_STEER = False
USE_NEURAL_FULL_CONTROL = False
USE_BC_DEBUG_ONLY = False
NEURAL_DRY_RUN = False
DEBUG_NEURAL_STEER = False
USE_SAFE_STEER_CORRECTION = False

BC_MODEL_DIR = Path("bc_model_1_51_85_output")
BC_MODEL_PATH = BC_MODEL_DIR / "best_bc_driver_model.keras"
BC_SCALER_PATH = BC_MODEL_DIR / "bc_scaler.pkl"
BC_FEATURES_PATH = BC_MODEL_DIR / "bc_feature_columns.json"

NEURAL_STEER_BLEND = 0.03
NEURAL_MAX_CORRECTION = 0.015

NEURAL_MIN_SPEED = 65.0
NEURAL_MIN_FRONT = 45.0
NEURAL_MAX_FRONT = 190.0
NEURAL_MAX_ANGLE = 0.14
NEURAL_MAX_TRACKPOS = 0.45
NEURAL_MAX_SPEEDZ = 2.2

# Neural debug runs only every N frames, otherwise TensorFlow may slow down control.
BC_DEBUG_EVERY_N_STEPS = 20


# =============================================================================
# 2. RUNTIME GLOBALS
# =============================================================================

# Control smoothing state
PREV_STEER = 0.0
PREV_ACCEL = 0.0
PREV_BRAKE = 0.0

# Track trend state
PREV_FRONT = None
FRONT_TREND = 0.0
FRONT_STABLE_COUNT = 0
CURRENT_FRONT_DELTA = 0.0

PREV_TRACK_POS = None
TRACK_POS_DELTA = 0.0

# Telemetry logging state
LOG_INITIALIZED = False
STEP_COUNTER = 0

PREV_LOG_FRONT = None
PREV_LOG_TRACK_POS = None

CURRENT_LAP = 0
LAP_STEP_COUNTER = 0
PREV_DIST_FROM_START = None

# Offline BC state logger state
BC_STATE_LOG_INITIALIZED = False
BC_STATE_LOG_COUNTER = 0

# Neural model state
BC_MODEL = None
BC_SCALER = None
BC_FEATURE_COLUMNS = None
BC_MODEL_LOAD_FAILED = False
BC_DEBUG_COUNTER = 0

# Control audit state
CONTROL_AUDIT_INITIALIZED = False
CONTROL_AUDIT_STEP = 0
LAST_CONTROL_TIME = None

LAST_PREDICT_MS = 0.0
LAST_NEURAL_CALLED = 0
LAST_NEURAL_APPLIED = 0
LAST_CONTROL_MODE = "RULE"

# Sector trace / planner state
SECTOR_TRACE_INITIALIZED = False
SECTOR_TRACE_STEP = 0

# Independent lap counter for sector_trace.csv.
# It is separate from LOG_FILE lap counter so sector tracing works even when ENABLE_LOGGING=False.
SECTOR_CURRENT_LAP = 0
SECTOR_LAP_STEP_COUNTER = 0
SECTOR_PREV_DIST_FROM_START = None


# =============================================================================
# 3. SMALL UTILITIES AND STATE UPDATE
# =============================================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def lerp(a, b, t):
    return a + (b - a) * clamp(t, 0.0, 1.0)


def smoothstep(edge0, edge1, x):
    if edge0 == edge1:
        return 0.0

    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def get_sector_id(S):
    dist = S.get("distFromStart", 0.0)
    if dist < 0:
        dist = 0.0

    sector_id = int(dist // SECTOR_SIZE_M)
    max_sector = int(TRACK_LENGTH_M // SECTOR_SIZE_M) + 1
    return max(0, min(sector_id, max_sector))


def calculate_dynamic_profile(S):
    """
    Adaptive profile used for analysis and optional future control.

    In Stage 1 this function is logged only. It does not change the car.
    Later we can blend its target speed into the live controller.
    """
    speed = S.get("speedX", 0.0)
    angle_abs = abs(S.get("angle", 0.0))
    track_pos_abs = abs(S.get("trackPos", 0.0))
    speed_z = S.get("speedZ", 0.0)
    track = S.get("track", [0.0] * 19)

    front = track[9]

    open_score = smoothstep(20.0, 180.0, front)
    close_score = 1.0 - open_score

    angle_risk = smoothstep(0.08, 0.32, angle_abs)
    edge_risk = smoothstep(0.55, 0.95, track_pos_abs)
    downhill_risk = smoothstep(1.5, 4.0, abs(speed_z)) if speed_z < -1.0 else 0.0

    front_opening = smoothstep(0.5, 4.0, FRONT_TREND)

    exit_confidence = front_opening
    exit_confidence *= 1.0 - smoothstep(0.18, 0.45, angle_abs)
    exit_confidence *= 1.0 - smoothstep(0.65, 0.95, track_pos_abs)
    exit_confidence = clamp(exit_confidence, 0.0, 1.0)

    risk = (
        close_score * 0.45
        + angle_risk * 0.25
        + edge_risk * 0.20
        + downhill_risk * 0.10
    )
    risk *= (1.0 - 0.45 * exit_confidence)
    risk = clamp(risk, 0.0, 1.0)

    # This is intentionally not over-scared: min corner speed starts at 72.
    dynamic_target_speed = lerp(TARGET_SPEED, 72.0, risk)
    dynamic_target_speed += 32.0 * exit_confidence

    if edge_risk > 0.75:
        dynamic_target_speed = min(dynamic_target_speed, 88.0)

    if angle_abs < 0.14 and track_pos_abs < 0.65 and front > 35:
        dynamic_target_speed = max(dynamic_target_speed, 105.0)

    brake_cap = lerp(0.08, 0.40, risk)
    brake_cap *= (1.0 - 0.65 * exit_confidence)
    brake_cap = clamp(brake_cap, 0.0, 0.40)

    accel_gain = lerp(0.45, 1.20, open_score)
    accel_gain += 0.35 * exit_confidence
    accel_gain *= (1.0 - 0.35 * edge_risk)
    accel_gain = clamp(accel_gain, 0.25, 1.35)

    speed_factor = smoothstep(80.0, 180.0, speed)
    steer_limit = lerp(0.68, 0.38, speed_factor)

    if front < 35 and speed < 95:
        steer_limit = max(steer_limit, 0.58)

    if edge_risk > 0.65:
        steer_limit = max(steer_limit, 0.62)

    steer_limit = clamp(steer_limit, 0.35, 0.70)

    return {
        "sector_id": get_sector_id(S),
        "risk": risk,
        "open_score": open_score,
        "exit_confidence": exit_confidence,
        "dynamic_target_speed": dynamic_target_speed,
        "brake_cap": brake_cap,
        "accel_gain": accel_gain,
        "steer_limit": steer_limit,
        "angle_risk": angle_risk,
        "edge_risk": edge_risk,
        "downhill_risk": downhill_risk,
    }


def calculate_apex_target_pos(S):
    # Placeholder kept because telemetry/training code expects this column.
    return 0.0


def update_front_trend(S):
    global PREV_FRONT
    global FRONT_TREND
    global FRONT_STABLE_COUNT
    global CURRENT_FRONT_DELTA
    global PREV_TRACK_POS
    global TRACK_POS_DELTA

    track = S.get("track", [0.0] * 19)
    front = track[9]
    track_pos = S.get("trackPos", 0.0)

    if PREV_FRONT is None:
        raw_delta = 0.0
    else:
        raw_delta = front - PREV_FRONT

    PREV_FRONT = front
    CURRENT_FRONT_DELTA = raw_delta

    FRONT_TREND = FRONT_TREND * 0.80 + raw_delta * 0.20

    if abs(raw_delta) < 4.0:
        FRONT_STABLE_COUNT += 1
    else:
        FRONT_STABLE_COUNT = 0

    if PREV_TRACK_POS is None:
        TRACK_POS_DELTA = 0.0
    else:
        TRACK_POS_DELTA = track_pos - PREV_TRACK_POS

    PREV_TRACK_POS = track_pos


def update_lap_counter(S):
    """
    Counts laps using distFromStart.

    lap = 0 -> start / out-lap part
    lap = 1 -> first full working lap
    lap = 2 -> second full working lap
    """
    global CURRENT_LAP
    global LAP_STEP_COUNTER
    global PREV_DIST_FROM_START

    dist_from_start = S.get("distFromStart", 0.0)
    speed = S.get("speedX", 0.0)

    if PREV_DIST_FROM_START is None:
        PREV_DIST_FROM_START = dist_from_start
        LAP_STEP_COUNTER = 0
        return CURRENT_LAP, LAP_STEP_COUNTER

    # New lap: distFromStart was large, then returned to small values.
    if PREV_DIST_FROM_START > 1000.0 and dist_from_start < 250.0 and speed > 20.0:
        CURRENT_LAP += 1
        LAP_STEP_COUNTER = 0
    else:
        LAP_STEP_COUNTER += 1

    PREV_DIST_FROM_START = dist_from_start

    return CURRENT_LAP, LAP_STEP_COUNTER


# =============================================================================
# 4. NEURAL MODEL FUNCTIONS
# =============================================================================

def load_bc_model_once():
    global BC_MODEL
    global BC_SCALER
    global BC_FEATURE_COLUMNS
    global BC_MODEL_LOAD_FAILED

    if BC_MODEL_LOAD_FAILED:
        return False

    if BC_MODEL is not None and BC_SCALER is not None and BC_FEATURE_COLUMNS is not None:
        return True

    try:
        print("[BC] Loading neural model...")

        BC_MODEL = tf.keras.models.load_model(BC_MODEL_PATH, compile=False)

        with open(BC_SCALER_PATH, "rb") as f:
            BC_SCALER = pickle.load(f)

        with open(BC_FEATURES_PATH, "r", encoding="utf-8") as f:
            BC_FEATURE_COLUMNS = json.load(f)

        if "lap_step" in BC_FEATURE_COLUMNS:
            raise RuntimeError(
                "This model uses lap_step. Do not use it online. "
                "Train/use a no-lapstep model."
            )

        print("[BC] Model loaded OK")
        print("[BC] Features:", BC_FEATURE_COLUMNS)

        return True

    except Exception as e:
        print("[BC] Model load failed:", e)
        BC_MODEL_LOAD_FAILED = True
        return False


def build_bc_feature_row_for_online(S):
    track = S.get("track", [0.0] * 19)

    row = {
        "distFromStart": S.get("distFromStart", 0.0),

        "speedX": S.get("speedX", 0.0),
        "speedY": S.get("speedY", 0.0),
        "speedZ": S.get("speedZ", 0.0),

        "angle": S.get("angle", 0.0),
        "trackPos": S.get("trackPos", 0.0),
        "trackPos_delta": TRACK_POS_DELTA,

        "z": S.get("z", 0.0),
        "rpm": S.get("rpm", 0.0),
        "gear": shift_gears(S),

        "front": track[9],
        "front_delta": CURRENT_FRONT_DELTA,
        "left_front": max(track[3:9]),
        "right_front": max(track[10:16]),
        "left_open": max(track[0:9]),
        "right_open": max(track[10:19]),
    }

    for i in range(19):
        row[f"track_{i}"] = track[i]

    return row


def predict_bc_action_fast(S):
    """
    Fast-ish TensorFlow call.

    Important:
    This is still risky inside TORCS control loop. Use debug/audit first.
    """
    if not load_bc_model_once():
        return None

    row = build_bc_feature_row_for_online(S)

    missing = [col for col in BC_FEATURE_COLUMNS if col not in row]
    if missing:
        print("[BC] Missing online features:", missing)
        return None

    x = pd.DataFrame([[row[col] for col in BC_FEATURE_COLUMNS]], columns=BC_FEATURE_COLUMNS)
    x_scaled = BC_SCALER.transform(x)
    x_scaled = np.asarray(x_scaled, dtype=np.float32)

    pred = BC_MODEL(x_scaled, training=False).numpy()[0]

    neural_steer = float(pred[0])
    neural_accel = float(pred[1])
    neural_brake = float(pred[2])

    return neural_steer, neural_accel, neural_brake


# =============================================================================
# 5. LOGGING AND CONTROL AUDIT
# =============================================================================

def reset_neural_runtime_marks():
    global LAST_PREDICT_MS
    global LAST_NEURAL_CALLED
    global LAST_NEURAL_APPLIED
    global LAST_CONTROL_MODE

    LAST_PREDICT_MS = 0.0
    LAST_NEURAL_CALLED = 0
    LAST_NEURAL_APPLIED = 0
    LAST_CONTROL_MODE = "RULE"


def mark_neural_called(predict_ms):
    global LAST_PREDICT_MS
    global LAST_NEURAL_CALLED
    global LAST_CONTROL_MODE

    LAST_PREDICT_MS = predict_ms
    LAST_NEURAL_CALLED = 1

    if LAST_CONTROL_MODE == "RULE":
        LAST_CONTROL_MODE = "NEURAL_DEBUG"


def mark_neural_applied():
    global LAST_NEURAL_APPLIED
    global LAST_CONTROL_MODE

    LAST_NEURAL_APPLIED = 1
    LAST_CONTROL_MODE = "NEURAL_APPLIED"


def log_telemetry(S, R, target_speed, target_brake):
    global LOG_INITIALIZED
    global STEP_COUNTER
    global PREV_LOG_FRONT
    global PREV_LOG_TRACK_POS

    track = S.get("track", [0.0] * 19)

    front = track[9]
    left_front = max(track[3:9])
    right_front = max(track[10:16])
    left_open = max(track[0:9])
    right_open = max(track[10:19])

    if PREV_LOG_FRONT is None:
        front_delta = 0.0
    else:
        front_delta = front - PREV_LOG_FRONT

    PREV_LOG_FRONT = front

    track_pos = S.get("trackPos", 0.0)

    if PREV_LOG_TRACK_POS is None:
        track_pos_delta = 0.0
    else:
        track_pos_delta = track_pos - PREV_LOG_TRACK_POS

    PREV_LOG_TRACK_POS = track_pos

    lap, lap_step = update_lap_counter(S)

    row = {
        "step": STEP_COUNTER,
        "lap": lap,
        "lap_step": lap_step,

        "distFromStart": S.get("distFromStart", 0.0),
        "distRaced": S.get("distRaced", 0.0),
        "curLapTime": S.get("curLapTime", 0.0),
        "lastLapTime": S.get("lastLapTime", 0.0),

        "speedX": S.get("speedX", 0.0),
        "speedY": S.get("speedY", 0.0),
        "speedZ": S.get("speedZ", 0.0),

        "angle": S.get("angle", 0.0),
        "trackPos": track_pos,
        "trackPos_delta": track_pos_delta,

        "z": S.get("z", 0.0),
        "rpm": S.get("rpm", 0.0),

        "gear": R.get("gear", 0),
        "steer": R.get("steer", 0.0),
        "accel": R.get("accel", 0.0),
        "brake": R.get("brake", 0.0),

        "target_speed": target_speed,
        "target_brake": target_brake,

        "front": front,
        "front_delta": front_delta,
        "left_front": left_front,
        "right_front": right_front,
        "left_open": left_open,
        "right_open": right_open,

        "apex_target": calculate_apex_target_pos(S),
    }

    for i in range(19):
        row[f"track_{i}"] = track[i]

    mode = "a" if LOG_INITIALIZED else "w"

    with open(LOG_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not LOG_INITIALIZED:
            writer.writeheader()
            LOG_INITIALIZED = True

        writer.writerow(row)

    STEP_COUNTER += 1


def log_bc_state_for_offline_compare(S, R, target_speed, target_brake):
    """
    Fast logger for offline BC comparison.

    Important:
    - does NOT load neural model
    - does NOT call model.predict()
    - does NOT change driving
    - only writes rule-based state/action into CSV
    """
    global BC_STATE_LOG_INITIALIZED
    global BC_STATE_LOG_COUNTER

    BC_STATE_LOG_COUNTER += 1

    if BC_STATE_LOG_COUNTER % BC_STATE_LOG_EVERY_N_STEPS != 0:
        return

    track = S.get("track", [0.0] * 19)

    row = {
        "step": BC_STATE_LOG_COUNTER,

        "distFromStart": S.get("distFromStart", 0.0),
        "distRaced": S.get("distRaced", 0.0),
        "curLapTime": S.get("curLapTime", 0.0),
        "lastLapTime": S.get("lastLapTime", 0.0),

        "speedX": S.get("speedX", 0.0),
        "speedY": S.get("speedY", 0.0),
        "speedZ": S.get("speedZ", 0.0),

        "angle": S.get("angle", 0.0),
        "trackPos": S.get("trackPos", 0.0),
        "trackPos_delta": TRACK_POS_DELTA,

        "z": S.get("z", 0.0),
        "rpm": S.get("rpm", 0.0),
        "gear": R.get("gear", 0),

        "front": track[9],
        "front_delta": CURRENT_FRONT_DELTA,
        "left_front": max(track[3:9]),
        "right_front": max(track[10:16]),
        "left_open": max(track[0:9]),
        "right_open": max(track[10:19]),

        "risk": profile["risk"],
        "open_score": profile["open_score"],
        "exit_confidence": profile["exit_confidence"],
        "dynamic_target_speed": profile["dynamic_target_speed"],
        "brake_cap": profile["brake_cap"],
        "accel_gain": profile["accel_gain"],

        "target_speed": target_speed,
        "target_brake": target_brake,

        "rule_steer": R.get("steer", 0.0),
        "rule_accel": R.get("accel", 0.0),
        "rule_brake": R.get("brake", 0.0),
    }

    for i in range(19):
        row[f"track_{i}"] = track[i]

    mode = "a" if BC_STATE_LOG_INITIALIZED else "w"

    with open(BC_STATE_LOG_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not BC_STATE_LOG_INITIALIZED:
            writer.writeheader()
            BC_STATE_LOG_INITIALIZED = True

        writer.writerow(row)


def log_control_audit(S, R, target_speed, target_brake):
    """
    Black-box logger for diagnosing:
    - hidden neural control
    - TensorFlow inference lag
    - control loop dt spikes
    - sudden steering/brake/gas events
    """
    global CONTROL_AUDIT_INITIALIZED
    global CONTROL_AUDIT_STEP
    global LAST_CONTROL_TIME

    if not ENABLE_CONTROL_AUDIT:
        return

    now = time.perf_counter()

    if LAST_CONTROL_TIME is None:
        dt_ms = 0.0
    else:
        dt_ms = (now - LAST_CONTROL_TIME) * 1000.0

    LAST_CONTROL_TIME = now

    track = S.get("track", [0.0] * 19)
    profile = calculate_dynamic_profile(S)

    row = {
        "step": CONTROL_AUDIT_STEP,
        "sector_id": profile["sector_id"],

        "mode": LAST_CONTROL_MODE,
        "neural_called": LAST_NEURAL_CALLED,
        "neural_applied": LAST_NEURAL_APPLIED,
        "predict_ms": LAST_PREDICT_MS,
        "control_dt_ms": dt_ms,

        "distFromStart": S.get("distFromStart", 0.0),
        "distRaced": S.get("distRaced", 0.0),
        "curLapTime": S.get("curLapTime", 0.0),
        "lastLapTime": S.get("lastLapTime", 0.0),

        "speedX": S.get("speedX", 0.0),
        "speedY": S.get("speedY", 0.0),
        "speedZ": S.get("speedZ", 0.0),

        "angle": S.get("angle", 0.0),
        "trackPos": S.get("trackPos", 0.0),

        "front": track[9],
        "left_open": max(track[0:9]),
        "right_open": max(track[10:19]),

        "risk": profile["risk"],
        "open_score": profile["open_score"],
        "exit_confidence": profile["exit_confidence"],
        "dynamic_target_speed": profile["dynamic_target_speed"],
        "brake_cap": profile["brake_cap"],
        "accel_gain": profile["accel_gain"],

        "target_speed": target_speed,
        "target_brake": target_brake,

        "final_steer": R.get("steer", 0.0),
        "final_accel": R.get("accel", 0.0),
        "final_brake": R.get("brake", 0.0),
        "final_gear": R.get("gear", 0),
    }

    mode = "a" if CONTROL_AUDIT_INITIALIZED else "w"

    with open(CONTROL_AUDIT_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not CONTROL_AUDIT_INITIALIZED:
            writer.writeheader()
            CONTROL_AUDIT_INITIALIZED = True

        writer.writerow(row)

    CONTROL_AUDIT_STEP += 1


def update_sector_trace_lap_counter(S):
    """
    Independent lap counter for sector_trace.csv.

    lap = 0 means launch / out-lap part.
    lap = 1, 2, 3... are completed working laps after crossing start line.
    """
    global SECTOR_CURRENT_LAP
    global SECTOR_LAP_STEP_COUNTER
    global SECTOR_PREV_DIST_FROM_START

    dist_from_start = S.get("distFromStart", 0.0)
    speed = S.get("speedX", 0.0)

    if SECTOR_PREV_DIST_FROM_START is None:
        SECTOR_PREV_DIST_FROM_START = dist_from_start
        SECTOR_LAP_STEP_COUNTER = 0
        return SECTOR_CURRENT_LAP, SECTOR_LAP_STEP_COUNTER

    # New lap: distFromStart was large, then wrapped back to the start zone.
    if SECTOR_PREV_DIST_FROM_START > 1000.0 and dist_from_start < 250.0 and speed > 20.0:
        SECTOR_CURRENT_LAP += 1
        SECTOR_LAP_STEP_COUNTER = 0
    else:
        SECTOR_LAP_STEP_COUNTER += 1

    SECTOR_PREV_DIST_FROM_START = dist_from_start

    return SECTOR_CURRENT_LAP, SECTOR_LAP_STEP_COUNTER


def log_sector_trace(S, R, target_speed, target_brake):
    """
    Stage-1 fast-path trace.

    This logger records sector-level information for building a track map.
    It must not change any control output.
    """
    global SECTOR_TRACE_INITIALIZED
    global SECTOR_TRACE_STEP

    if not ENABLE_SECTOR_TRACE:
        return

    track = S.get("track", [0.0] * 19)
    wheel_spin = S.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])
    rear_slip = (wheel_spin[2] + wheel_spin[3]) - (wheel_spin[0] + wheel_spin[1])
    profile = calculate_dynamic_profile(S)
    lap, lap_step = update_sector_trace_lap_counter(S)

    row = {
        "step": SECTOR_TRACE_STEP,
        "lap": lap,
        "lap_step": lap_step,
        "sector_id": profile["sector_id"],
        "sector_start_m": profile["sector_id"] * SECTOR_SIZE_M,
        "sector_end_m": (profile["sector_id"] + 1) * SECTOR_SIZE_M,

        "distFromStart": S.get("distFromStart", 0.0),
        "distRaced": S.get("distRaced", 0.0),
        "curLapTime": S.get("curLapTime", 0.0),
        "lastLapTime": S.get("lastLapTime", 0.0),

        "speedX": S.get("speedX", 0.0),
        "speedY": S.get("speedY", 0.0),
        "speedZ": S.get("speedZ", 0.0),
        "angle": S.get("angle", 0.0),
        "trackPos": S.get("trackPos", 0.0),
        "trackPos_delta": TRACK_POS_DELTA,
        "front": track[9],
        "front_delta": CURRENT_FRONT_DELTA,
        "front_trend": FRONT_TREND,
        "left_open": max(track[0:9]),
        "right_open": max(track[10:19]),
        "rear_slip": rear_slip,

        "risk": profile["risk"],
        "open_score": profile["open_score"],
        "exit_confidence": profile["exit_confidence"],
        "dynamic_target_speed": profile["dynamic_target_speed"],
        "brake_cap": profile["brake_cap"],
        "accel_gain": profile["accel_gain"],
        "steer_limit": profile["steer_limit"],

        "target_speed": target_speed,
        "target_brake": target_brake,
        "final_steer": R.get("steer", 0.0),
        "final_accel": R.get("accel", 0.0),
        "final_brake": R.get("brake", 0.0),
        "final_gear": R.get("gear", 0),
    }

    mode = "a" if SECTOR_TRACE_INITIALIZED else "w"

    with open(SECTOR_TRACE_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not SECTOR_TRACE_INITIALIZED:
            writer.writeheader()
            SECTOR_TRACE_INITIALIZED = True

        writer.writerow(row)

    SECTOR_TRACE_STEP += 1


def finish_frame_logging(S, R, target_brake):
    """
    One common logging exit point.

    This prevents losing rows when drive_modular exits early
    from safety / anti-slide / recovery branches.
    """
    current_target_speed = calculate_target_speed(S)

    if ENABLE_LOGGING:
        log_telemetry(S, R, current_target_speed, target_brake)

    if ENABLE_BC_STATE_LOG:
        log_bc_state_for_offline_compare(S, R, current_target_speed, target_brake)

    if ENABLE_SECTOR_TRACE:
        log_sector_trace(S, R, current_target_speed, target_brake)

    if ENABLE_CONTROL_AUDIT:
        log_control_audit(S, R, current_target_speed, target_brake)


def load_sector_speed_profile_once():
    global SECTOR_SPEED_PROFILE

    if SECTOR_SPEED_PROFILE is not None:
        return SECTOR_SPEED_PROFILE

    SECTOR_SPEED_PROFILE = {}

    if not ENABLE_SECTOR_SPEED_PROFILE:
        return SECTOR_SPEED_PROFILE

    path = Path(SECTOR_PROFILE_FILE)
    if not path.exists():
        print(f"[SECTOR] Profile file not found: {SECTOR_PROFILE_FILE}")
        return SECTOR_SPEED_PROFILE

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Expected format: {"12": {"speed_bonus": 4.0, "brake_scale": 0.95}, ...}
        SECTOR_SPEED_PROFILE = {int(k): v for k, v in raw.items()}
        print("[SECTOR] Loaded sectors:", len(SECTOR_SPEED_PROFILE))
    except Exception as e:
        print("[SECTOR] Failed to load sector profile:", e)
        SECTOR_SPEED_PROFILE = {}


    return SECTOR_SPEED_PROFILE


def load_track_plan_once():
    global TRACK_PLAN

    if TRACK_PLAN is not None:
        return TRACK_PLAN

    TRACK_PLAN = {}

    if not ENABLE_TRACK_PLAN:
        return TRACK_PLAN

    path = Path(TRACK_PLAN_FILE)
    if not path.exists():
        print(f"[PLANNER] Track plan file not found: {TRACK_PLAN_FILE}")
        return TRACK_PLAN

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # New format: {"_meta": {...}, "sectors": {"12": {...}}}
        sectors = raw.get("sectors", raw) if isinstance(raw, dict) else {}
        TRACK_PLAN = {int(k): v for k, v in sectors.items() if str(k).isdigit()}
        print("[PLANNER] Loaded plan sectors:", len(TRACK_PLAN))
    except Exception as e:
        print("[PLANNER] Failed to load track plan:", e)
        TRACK_PLAN = {}

    return TRACK_PLAN


def get_track_plan_cfg_by_dist(dist):
    if not ENABLE_TRACK_PLAN:
        return {}

    plan = load_track_plan_once()
    if not plan:
        return {}

    if dist < 0:
        dist = 0.0

    # Dist can exceed track length near finish; normalize to current lap.
    dist = float(dist) % TRACK_LENGTH_M
    sector_id = int(dist // SECTOR_SIZE_M)
    max_sector = int(TRACK_LENGTH_M // SECTOR_SIZE_M) + 1
    sector_id = max(0, min(sector_id, max_sector))
    return plan.get(sector_id, {})


def get_track_plan_cfg(S, offset_m=0.0):
    dist = float(S.get("distFromStart", 0.0)) + float(offset_m)
    return get_track_plan_cfg_by_dist(dist)


def get_track_planned_speed(S, offset_m=0.0):
    cfg = get_track_plan_cfg(S, offset_m)
    if not cfg:
        return None
    value = cfg.get("planned_speed", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def calculate_preview_brake(S):
    """
    Planner V1 preview braking.

    The old driver mostly reacts to current front sensor. This function looks
    ahead in the track plan and starts braking if a slower planned sector is
    coming. It is conservative and lets legacy emergency braking handle current
    danger.
    """
    if not PREVIEW_BRAKE_ENABLED or not ENABLE_TRACK_PLAN:
        return 0.0

    speed_kmh = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z = float(S.get("speedZ", 0.0))
    track = S.get("track", [0.0] * 19)
    front = float(track[9])

    if speed_kmh < 70.0:
        return 0.0
    if front < 34.0:
        return 0.0
    if angle_abs > 0.26 or track_pos_abs > 0.76:
        return 0.0
    if speed_z < -4.5 and front < 55.0:
        return 0.0

    best_brake = 0.0

    for lookahead_m in PREVIEW_LOOKAHEAD_M:
        planned = get_track_planned_speed(S, lookahead_m)
        if planned is None:
            continue

        # Small margin prevents useless micro-brakes.
        target_kmh = planned + 5.0
        if speed_kmh <= target_kmh:
            continue

        v_now = speed_kmh / 3.6
        v_target = target_kmh / 3.6
        d = max(float(lookahead_m), 20.0)
        required_decel = max(0.0, (v_now * v_now - v_target * v_target) / (2.0 * d))

        # Convert rough required deceleration to TORCS brake command.
        brake = required_decel / 8.0

        # Planner should prepare braking, not panic-brake.
        if lookahead_m >= 175.0:
            brake *= 0.72
        elif lookahead_m >= 125.0:
            brake *= 0.86

        best_brake = max(best_brake, brake)

    if best_brake < 0.035:
        return 0.0

    # Current openness allows a little more preview braking on fast straights.
    if front > 115.0 and angle_abs < 0.08:
        return clamp(best_brake, 0.0, 0.34)

    return clamp(best_brake, 0.0, 0.26)


def get_sector_cfg(S):
    """
    Returns merged sector config for current distFromStart.

    Priority: legacy sector_profile_active.json first, track_plan_active.json
    overrides only when it provides the same key. This lets us keep old safe
    profile logic while adding planner fields.
    """
    merged = {}

    if ENABLE_SECTOR_SPEED_PROFILE:
        profile = load_sector_speed_profile_once()
        sector_id = get_sector_id(S)
        base_cfg = profile.get(sector_id, {})
        if isinstance(base_cfg, dict):
            merged.update(base_cfg)

    if ENABLE_TRACK_PLAN:
        plan_cfg = get_track_plan_cfg(S)
        if isinstance(plan_cfg, dict):
            merged.update(plan_cfg)

    return merged


def apply_sector_brake_scale(S, brake_value):
    """
    Scales brake only in selected sectors.
    This does not remove hard safety completely; it only softens commanded brake.
    """
    brake_value = float(brake_value)

    if brake_value <= 0.0:
        return 0.0

    cfg = get_sector_cfg(S)
    scale = float(cfg.get("brake_scale", 1.0))

    return clamp(brake_value * scale, 0.0, 1.0)


def apply_sector_accel_bias(S, accel_value):
    """
    Adds small acceleration bias in selected sectors.
    Hard safety returns 0.0 stay 0.0.
    """
    accel_value = float(accel_value)

    if accel_value <= 0.0:
        return 0.0

    cfg = get_sector_cfg(S)
    bias = float(cfg.get("accel_bias", 0.0))

    return clamp(accel_value + bias, 0.0, 1.0)


def is_sector_floor_safe(S):
    """
    Safety gate for experimental sector acceleration floors.
    This prevents min_accel from overriding real danger states.
    """
    track = S.get("track", [0.0] * 19)
    front = track[9]
    speed = S.get("speedX", 0.0)
    angle_abs = abs(S.get("angle", 0.0))
    track_pos_abs = abs(S.get("trackPos", 0.0))
    speed_z = S.get("speedZ", 0.0)

    if front < 14.0:
        return False

    if track_pos_abs > 0.62:
        return False

    if angle_abs > 0.24:
        return False

    if speed_z < -4.2 and front < 28.0:
        return False

    if speed > 135.0 and front < 35.0:
        return False

    return True


def apply_sector_final_brake_cap(S, brake_value):
    """
    Caps final smoothed brake in selected sectors.

    Why this exists:
    apply_sector_brake_scale changes target brake, but smooth_brake can still
    carry previous braking into the exit. That costs time in sectors where the
    car is already aligned and should accelerate.
    """
    brake_value = float(brake_value)

    if brake_value <= 0.0:
        return 0.0

    cfg = get_sector_cfg(S)

    if "final_brake_cap" not in cfg:
        return brake_value

    cap = float(cfg.get("final_brake_cap", 1.0))

    track = S.get("track", [0.0] * 19)
    front = track[9]
    angle_abs = abs(S.get("angle", 0.0))
    track_pos_abs = abs(S.get("trackPos", 0.0))

    # Do not cap brake in true emergency.
    if front < 12.0 or angle_abs > 0.30 or track_pos_abs > 0.72:
        return brake_value

    return clamp(brake_value, 0.0, cap)


def apply_sector_final_accel_floor(S, accel_value, brake_value):
    """
    Applies minimum throttle in selected exit sectors.

    This is not a global aggression increase. It only works when:
    - the profile explicitly asks for min_accel
    - brake is not active
    - the safety gate says the car is aligned enough
    """
    accel_value = float(accel_value)
    brake_value = float(brake_value)

    cfg = get_sector_cfg(S)

    if "min_accel" not in cfg:
        return clamp(accel_value, 0.0, 1.0)

    if brake_value > 0.14:
        return clamp(accel_value, 0.0, 1.0)

    if not is_sector_floor_safe(S):
        return clamp(accel_value, 0.0, 1.0)

    min_accel = float(cfg.get("min_accel", 0.0))

    return clamp(max(accel_value, min_accel), 0.0, 1.0)





def v23_in_any_range(dist, ranges):
    dist = float(dist) % TRACK_LENGTH_M
    for r in ranges:
        if len(r) < 2:
            continue
        a, b = float(r[0]), float(r[1])
        if a <= dist < b:
            return True
    return False



V23_TURNIN_RANGES = [
    (500.0, 555.0, +1.0),
    (775.0, 815.0, -1.0),
    (1050.0, 1090.0, -1.0),
    (1575.0, 1615.0, +1.0),
    (2470.0, 2535.0, -1.0),
    (2630.0, 2705.0, +1.0),
    (3170.0, 3265.0, +1.0),
]

V23_EXIT_RANGES = [
    (225.0, 275.0),
    (525.0, 565.0),
    (800.0, 875.0),
    (1075.0, 1135.0),
    (1600.0, 1665.0),
    (2500.0, 2550.0),
    (2700.0, 2835.0),
    (3025.0, 3085.0),
    (3325.0, 3385.0),
]

def apply_v23_apex_steer_assist(S, steer):
    """
    V23 local apex logic.

    This is deliberately NOT full AI steering and NOT global line assist.
    It only helps the existing steering reach the already-needed rotation earlier
    in known turn-in zones, then unwinds slightly on exits.
    """
    if V23_APEX_STEER_GAIN <= 0.0:
        return steer

    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9])

    if speed < 55.0 or front < 18.0 or angle_abs > 0.32 or track_pos_abs > 0.82:
        return steer

    assist = 0.0
    for start, end, direction in V23_TURNIN_RANGES:
        if start <= dist < end:
            local = V23_APEX_STEER_GAIN
            if speed > 150.0:
                local *= 0.65
            elif speed > 125.0:
                local *= 0.82
            if front < 35.0:
                local *= 1.20
            assist += direction * local
            break

    if assist != 0.0:
        # Do not push farther if already very close to the edge.
        if S.get("trackPos", 0.0) * assist > 0.0 and track_pos_abs > 0.55:
            assist *= 0.45
        steer = steer + assist

    # Exit unwind: if front opens and car is aligned, reduce residual steering.
    if V23_EXIT_UNWIND > 0.0 and v23_in_any_range(dist, V23_EXIT_RANGES):
        if front > 55.0 and angle_abs < 0.12 and track_pos_abs < 0.72 and speed > 70.0:
            steer = lerp(steer, 0.0, V23_EXIT_UNWIND)

    return clamp(steer, -0.70, 0.70)


def apply_v23_power_commit(S, accel_value, brake_value):
    """
    V23 local acceleration layer.

    Goal: raise exit acceleration and top speed without changing corner braking.
    It activates only when brake is nearly absent and the car is reasonably aligned.
    """
    accel_value = float(accel_value)
    brake_value = float(brake_value)

    if brake_value > 0.085:
        return clamp(accel_value, 0.0, 1.0)

    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z = float(S.get("speedZ", 0.0))
    track = S.get("track", [0.0] * 19)
    front = float(track[9])

    if abs(speed_z) > 4.0 or track_pos_abs > 0.82:
        return clamp(accel_value, 0.0, 1.0)

    # Very clean straight: force full acceleration up to the new top-speed target.
    if front > 155.0 and angle_abs < 0.060 and track_pos_abs < 0.48 and speed < TARGET_SPEED - 1.0:
        return clamp(max(accel_value, V23_STRAIGHT_MIN_ACCEL), 0.0, 1.0)

    # Known exit zones: commit to throttle earlier if the car is safe enough.
    if v23_in_any_range(dist, V23_EXIT_RANGES):
        if front > 38.0 and angle_abs < V23_EXIT_MAX_ANGLE and track_pos_abs < 0.76 and FRONT_TREND > -1.2:
            if speed < 155.0:
                return clamp(max(accel_value, V23_EXIT_MIN_ACCEL), 0.0, 1.0)
            if front > 110.0 and angle_abs < 0.10 and speed < TARGET_SPEED - 8.0:
                return clamp(max(accel_value, 0.94), 0.0, 1.0)

    return clamp(accel_value, 0.0, 1.0)


def shift_gears(S):
    """
    V23 RPM-based shifting with speed fallback.

    The old speed-only shifting worked, but it can hold a gear too long/short
    when acceleration changes. This version uses RPM first, speed second.
    """
    speed = float(S.get("speedX", 0.0))
    rpm = float(S.get("rpm", 0.0))
    gear = int(S.get("gear", 1) or 1)

    if gear < 1 or gear > 6:
        gear = 1

    # Speed fallback / sanity gear. Keeps the car from staying in a wrong gear.
    if speed < 30.0:
        speed_gear = 1
    elif speed < 55.0:
        speed_gear = 2
    elif speed < 90.0:
        speed_gear = 3
    elif speed < 135.0:
        speed_gear = 4
    elif speed < 200.0:
        speed_gear = 5
    else:
        speed_gear = 6

    if rpm < 500.0:
        return speed_gear

    # Prevent huge mismatch between sensor gear and sensible speed gear.
    if gear < speed_gear - 1:
        return gear + 1
    if gear > speed_gear + 1:
        return gear - 1

    min_speed_for_upshift = {1: 25.0, 2: 48.0, 3: 82.0, 4: 122.0, 5: 175.0}
    max_speed_for_downshift = {2: 38.0, 3: 70.0, 4: 108.0, 5: 150.0, 6: 188.0}

    if gear < 6 and rpm > V23_UPSHIFT_RPM and speed > min_speed_for_upshift.get(gear, 0.0):
        return gear + 1

    if gear > 1 and rpm < V23_DOWNSHIFT_RPM and speed < max_speed_for_downshift.get(gear, 999.0):
        return gear - 1

    return gear




def v35_progress(dist, start, end):
    if end <= start:
        return 1.0
    return clamp((float(dist) - float(start)) / (float(end) - float(start)), 0.0, 1.0)


def apply_v35_centered_turnin_steer(S, steer):
    """
    Local earlier turn-in without the V32 wiggle problem.

    This does not globally make steering faster. It only adds a small pre-load
    toward the safe/right side in the three complexes where the logs showed the
    car sitting on the left edge. The base smooth_steer still limits step size.
    """
    if not V35_CENTERED_CORNER_ENABLED:
        return clamp(float(steer), -0.70, 0.70)

    steer = float(steer)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos = float(S.get("trackPos", 0.0))
    track_pos_abs = abs(track_pos)
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if speed < 45.0 or angle_abs > 0.42 or track_pos_abs > 1.02:
        return clamp(steer, -0.70, 0.70)

    assist = 0.0

    # All three observed edge-risk complexes are right-turn / left-edge cases:
    # positive steer moves the car back toward the middle/safe side.
    # (start, early_end, apex_end, direction, desired max assist scale)
    zones = [
        (382.0, 418.0, 466.0, +1.0, 0.82),
        (1450.0, 1505.0, 1565.0, +1.0, 1.00),
        (1825.0, 1888.0, 1950.0, +1.0, 1.22),
    ]

    for start, early_end, apex_end, direction, scale in zones:
        if start <= dist < apex_end:
            if dist < early_end:
                p = v35_progress(dist, start, early_end)
                local = V35_TURNIN_GAIN * (0.35 + 0.65 * p) * scale
            else:
                # Mid-corner: keep steering help only if the car is still too far left.
                p = 1.0 - 0.45 * v35_progress(dist, early_end, apex_end)
                local = V35_TURNIN_GAIN * p * scale

            # More correction if it is actually near the left edge; much less if centered.
            if track_pos < -0.70:
                local *= 1.55
            elif track_pos < -0.50:
                local *= 1.35
            elif track_pos < -0.28:
                local *= 1.10
            elif track_pos < -0.10:
                local *= 0.70
            else:
                local *= 0.32

            # Do not force rotation if road is too closed or speed is too high.
            if front < 13.0 and speed > 105.0:
                local *= 0.65
            if speed > 185.0:
                local *= 0.72
            elif speed > 160.0:
                local *= 0.84

            assist = direction * local
            break

    if assist != 0.0:
        # Anti-wiggle: never push against an already large opposite steering input.
        if steer * assist < 0.0 and abs(steer) > 0.32:
            assist *= 0.35
        steer += assist

    return clamp(steer, -0.62, 0.62)


def apply_v35_centered_corner_brake(S, brake_value):
    """
    Stronger local brake so the car reaches the middle of the turn instead of
    arriving wide and surviving on the outside edge.
    """
    if not V35_CENTERED_CORNER_ENABLED:
        return clamp(float(brake_value), 0.0, 1.0)

    brake_value = float(brake_value)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos = float(S.get("trackPos", 0.0))
    track_pos_abs = abs(track_pos)
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if track_pos_abs > 1.05 or angle_abs > 0.55:
        return clamp(max(brake_value, 0.30), 0.0, 0.70)

    floor = 0.0

    # 400-475m: logs show left-edge peaks up to about 0.90 on clean laps.
    if 372.0 <= dist < 405.0 and speed > 150.0:
        floor = max(floor, 0.12)
    if 405.0 <= dist < 438.0 and speed > 112.0:
        floor = max(floor, 0.22)
    if 438.0 <= dist < 468.0:
        if speed > 82.0:
            floor = max(floor, 0.30)
        if track_pos < -0.62 and speed > 68.0:
            floor = max(floor, 0.36)

    # 1500-1575m: stable laps were already at ~0.94-0.96 abs(trackPos).
    if 1440.0 <= dist < 1490.0 and speed > 172.0:
        floor = max(floor, 0.16)
    if 1490.0 <= dist < 1528.0 and speed > 128.0:
        floor = max(floor, 0.34)
    if 1528.0 <= dist < 1568.0:
        if speed > 88.0:
            floor = max(floor, 0.36)
        if track_pos < -0.72 and speed > 74.0:
            floor = max(floor, 0.44)

    if 1775.0 <= dist < 1825.0 and speed > 220.0:
        floor = max(floor, 0.10)
    if 1825.0 <= dist < 1868.0 and speed > 190.0:
        floor = max(floor, 0.22)
    if 1868.0 <= dist < 1908.0 and speed > 138.0:
        floor = max(floor, 0.42)
    if 1908.0 <= dist < 1948.0:
        if speed > 88.0:
            floor = max(floor, 0.54)
        if track_pos < -0.62 and speed > 72.0:
            floor = max(floor, 0.60)
        if track_pos < -0.82 and speed > 60.0:
            floor = max(floor, 0.66)

    # Do not drag the car when it is already centered, slow enough, and opening road.
    if floor > 0.0:
        if track_pos_abs < 0.38 and angle_abs < 0.080 and front > 55.0:
            floor *= 0.55
        floor *= V35_BRAKE_STRENGTH
        return clamp(max(brake_value, floor), 0.0, V35_BRAKE_CAP)

    return clamp(brake_value, 0.0, 1.0)


def apply_v35_midcorner_gas_commit(S, accel_value, brake_value):
    """
    Open throttle earlier in the corner only AFTER the car is centered enough.
    This deliberately does not override braking or edge guard.
    """
    if not V35_CENTERED_CORNER_ENABLED:
        return clamp(float(accel_value), 0.0, 1.0)

    accel_value = float(accel_value)
    brake_value = float(brake_value)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z = abs(float(S.get("speedZ", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if brake_value > 0.10 or track_pos_abs > 0.78 or angle_abs > 0.27 or speed_z > 4.4:
        return clamp(accel_value, 0.0, 1.0)

    # Once centered in these corners, stop being lazy with throttle.
    in_corner_commit = (
        (448.0 <= dist < 520.0 and front > 18.0 and 52.0 <= speed < 112.0) or
        (1548.0 <= dist < 1628.0 and front > 18.0 and 54.0 <= speed < 122.0) or
        (1940.0 <= dist < 2012.0 and front > 18.0 and 52.0 <= speed < 126.0)
    )

    if in_corner_commit:
        floor = V35_MIDCORNER_MIN_ACCEL
        if front > 45.0 and angle_abs < 0.16 and track_pos_abs < 0.60:
            floor = max(floor, V35_EXIT_MIN_ACCEL)
        if front > 70.0 and angle_abs < 0.11 and track_pos_abs < 0.52:
            floor = 1.0
        return clamp(max(accel_value, floor), 0.0, 1.0)

    return clamp(accel_value, 0.0, 1.0)






V36_EDGE_APEX_ENABLED = True
V39_PROFILE_NAME = "carry"
V39_SHORT_BRAKE_MULT = 0.96
V39_TURNIN_PRELOAD_GAIN = 0.01
V39_RELEASE_BRAKE_CAP = 0.075
V39_ENTRY_BRAKE_CAP = 0.58
V39_CORNER_ACCEL_FLOOR = 0.88
V39_EXIT_ACCEL_FLOOR = 0.96
V39_BRAKE_KILL_ACCEL_THRESHOLD = 0.3

# (entry_start, pre_start, apex_start, exit_start, exit_end, steer_direction)
V36_EDGE_APEX_ZONES = [
    (355.0, 402.0, 438.0, 458.0, 530.0, +1.0),
    (1415.0, 1485.0, 1532.0, 1560.0, 1640.0, +1.0),
    (1765.0, 1845.0, 1920.0, 1952.0, 2025.0, +1.0),
]

def v36_zone_progress(dist, a, b):
    if b <= a:
        return 1.0
    return clamp((float(dist) - float(a)) / (float(b) - float(a)), 0.0, 1.0)


def apply_v36_edge_apex_turnin(S, steer):
    """Subtle early preload only; no aggressive steering assist."""
    if not V36_EDGE_APEX_ENABLED:
        return clamp(float(steer), -0.70, 0.70)

    steer = float(steer)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos = float(S.get("trackPos", 0.0))
    track_pos_abs = abs(track_pos)
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if speed < 45.0 or speed > 230.0 or angle_abs > 0.36 or track_pos_abs > 1.00:
        return clamp(steer, -0.70, 0.70)

    assist = 0.0
    for entry_start, pre_start, apex_start, exit_start, exit_end, direction in V36_EDGE_APEX_ZONES:
        if entry_start <= dist < exit_start:
            if dist < pre_start:
                p = v36_zone_progress(dist, entry_start, pre_start)
                local = V39_TURNIN_PRELOAD_GAIN * (0.18 + 0.62 * p)
            elif dist < apex_start:
                p = v36_zone_progress(dist, pre_start, apex_start)
                local = V39_TURNIN_PRELOAD_GAIN * (0.78 + 0.25 * p)
            else:
                p = 1.0 - 0.65 * v36_zone_progress(dist, apex_start, exit_start)
                local = V39_TURNIN_PRELOAD_GAIN * p

            # Add more only when the car is still outside. If it already crossed
            # toward the opposite edge, stop pushing it further.
            if track_pos < -0.82:
                local *= 1.35
            elif track_pos < -0.60:
                local *= 1.15
            elif track_pos < -0.35:
                local *= 0.92
            elif track_pos < 0.10:
                local *= 0.55
            else:
                local *= 0.18

            # Avoid snap while road is closed or speed is still high.
            if front < 10.0 and speed > 95.0:
                local *= 0.55
            if speed > 170.0:
                local *= 0.62
            elif speed > 135.0:
                local *= 0.78

            assist = direction * local
            break

    if assist != 0.0:
        if steer * assist < 0.0:
            assist *= 0.18
        if abs(steer) > 0.50 and steer * assist > 0.0:
            assist *= 0.28
        steer += assist

    return clamp(steer, -0.62, 0.62)


def apply_v36_edge_apex_brake_phase(S, brake_value):
    """
    Short decisive brake near the entry, then release. This avoids V38's issue:
    early brake -> turn -> continued speed bleed.
    """
    if not V36_EDGE_APEX_ENABLED:
        return clamp(float(brake_value), 0.0, 1.0)

    brake_value = float(brake_value)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos = float(S.get("trackPos", 0.0))
    track_pos_abs = abs(track_pos)
    speed_z = abs(float(S.get("speedZ", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    # True off-track / almost off-track still needs real rescue.
    if track_pos_abs > 1.03:
        return clamp(max(brake_value, 0.34), 0.0, 0.78)

    floor = 0.0

    if 365.0 <= dist < 405.0 and speed > 166.0:
        floor = max(floor, 0.08)
    if 405.0 <= dist < 438.0 and speed > 126.0:
        floor = max(floor, 0.22)
    if 438.0 <= dist < 456.0 and speed > 84.0:
        floor = max(floor, 0.30)

    # Zone 2.
    if 1425.0 <= dist < 1488.0 and speed > 188.0:
        floor = max(floor, 0.09)
    if 1488.0 <= dist < 1534.0 and speed > 136.0:
        floor = max(floor, 0.28)
    if 1534.0 <= dist < 1560.0 and speed > 88.0:
        floor = max(floor, 0.34)

    # Zone 3 / sector 77 risk complex. Keep the guard, but shorter.
    if 1775.0 <= dist < 1848.0 and speed > 220.0:
        floor = max(floor, 0.10)
    if 1848.0 <= dist < 1918.0 and speed > 142.0:
        floor = max(floor, 0.36)
    if 1918.0 <= dist < 1950.0 and speed > 88.0:
        floor = max(floor, 0.42)

    if floor > 0.0:
        # More brake only if still hot AND near the dangerous outside side.
        if track_pos_abs > 0.78:
            floor *= 1.12
        elif track_pos_abs < 0.52 and front > 18.0:
            floor *= 0.70
        if angle_abs > 0.27 and speed < 115.0:
            floor *= 0.70
        if speed_z > 5.5:
            floor *= 0.85
        brake_value = max(brake_value, floor * V39_SHORT_BRAKE_MULT)

    mid_corner = (
        (456.0 <= dist < 530.0) or
        (1560.0 <= dist < 1640.0) or
        (1950.0 <= dist < 2025.0)
    )
    near_apex = (
        (438.0 <= dist < 476.0) or
        (1534.0 <= dist < 1588.0) or
        (1918.0 <= dist < 1978.0)
    )

    if near_apex:
        # If speed is already low enough, do not keep braking while steering.
        low_enough = (
            (438.0 <= dist < 476.0 and speed < 102.0) or
            (1534.0 <= dist < 1588.0 and speed < 116.0) or
            (1918.0 <= dist < 1978.0 and speed < 118.0)
        )
        if low_enough and track_pos_abs < 0.92 and angle_abs < 0.34:
            brake_value = min(brake_value, V39_RELEASE_BRAKE_CAP * 1.25)

    if mid_corner:
        if track_pos_abs < 0.90 and angle_abs < 0.33 and front > 9.0:
            brake_value = min(brake_value, V39_RELEASE_BRAKE_CAP)
        if track_pos_abs < 0.72 and angle_abs < 0.26 and front > 14.0:
            brake_value = min(brake_value, V39_RELEASE_BRAKE_CAP * 0.55)
        if track_pos_abs < 0.58 and angle_abs < 0.20 and front > 24.0:
            brake_value = min(brake_value, 0.0)

    return clamp(brake_value, 0.0, V39_ENTRY_BRAKE_CAP)


def apply_v36_edge_apex_speed_commit(S, accel_value, brake_value):
    """Carry speed through the hairpin once brake has been released."""
    if not V36_EDGE_APEX_ENABLED:
        return clamp(float(accel_value), 0.0, 1.0)

    accel_value = float(accel_value)
    brake_value = float(brake_value)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z = abs(float(S.get("speedZ", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if track_pos_abs > 0.94 or angle_abs > 0.37 or speed_z > 6.2:
        return clamp(accel_value, 0.0, 1.0)

    in_corner = (
        (452.0 <= dist < 530.0 and 62.0 <= speed < 126.0 and front > 8.0) or
        (1555.0 <= dist < 1640.0 and 64.0 <= speed < 142.0 and front > 9.0) or
        (1950.0 <= dist < 2025.0 and 64.0 <= speed < 146.0 and front > 9.0)
    )
    if not in_corner:
        return clamp(accel_value, 0.0, 1.0)

    # If brake is still meaningful, first let brake release; do not fight brake
    # and gas. A tiny residual brake is tolerated for transition.
    if brake_value > 0.18:
        return clamp(accel_value, 0.0, 1.0)

    floor = V39_CORNER_ACCEL_FLOOR
    if track_pos_abs < 0.82 and angle_abs < 0.30 and front > 12.0:
        floor = max(floor, V39_EXIT_ACCEL_FLOOR)
    if track_pos_abs < 0.68 and angle_abs < 0.24 and front > 18.0:
        floor = max(floor, 0.98)
    if track_pos_abs < 0.56 and angle_abs < 0.18 and front > 28.0:
        floor = 1.0

    # If there is still small residual brake, keep it progressive.
    if brake_value > 0.08:
        floor = min(floor, 0.86)

    return clamp(max(accel_value, floor), 0.0, 1.0)

# =============================================================================
# 6. RULE-BASED CONTROL FUNCTIONS
# =============================================================================

def smooth_steer(target_steer, S):
    global PREV_STEER

    speed = S.get("speedX", 0.0)
    track = S.get("track", [0.0] * 19)
    front = track[9]
    track_pos = S.get("trackPos", 0.0)

    target_steer = clamp(target_steer, -0.70, 0.70)
    diff = target_steer - PREV_STEER

    if speed > 150:
        max_step = 0.035
    elif speed > 120:
        max_step = 0.045
    elif speed > 80:
        max_step = 0.060
    else:
        max_step = 0.075

    if front < 25 and speed < 90:
        max_step = 0.095

    if front < 15 and speed < 80:
        max_step = 0.110

    if abs(track_pos) > 0.65:
        max_step += 0.025

    diff = clamp(diff, -max_step, max_step)

    steer = PREV_STEER + diff
    PREV_STEER = steer

    return clamp(steer, -0.70, 0.70)


def calculate_steering(S):
    angle = S.get("angle", 0.0)
    track_pos = S.get("trackPos", 0.0)
    speed = S.get("speedX", 0.0)
    track = S.get("track", [0.0] * 19)

    front = track[9]

    angle_correction = angle * STEER_GAIN / math.pi
    position_correction = track_pos * CENTERING_GAIN

    steer = angle_correction - position_correction
    steer = apply_final_turn_steer_assist(S, steer)
    steer = apply_v23_apex_steer_assist(S, steer)
    steer = apply_v35_centered_turnin_steer(S, steer)
    steer = apply_v36_edge_apex_turnin(S, steer)

    # Tight corner rescue.
    if front < 25 and speed < 90:
        if track_pos < -0.85:
            steer += 0.45
        elif track_pos < -0.70:
            steer += 0.36
        elif track_pos < -0.55:
            steer += 0.28
        elif track_pos < -0.35:
            steer += 0.20
        elif track_pos < -0.20:
            steer += 0.12

        if track_pos > 0.85:
            steer -= 0.45
        elif track_pos > 0.70:
            steer -= 0.36
        elif track_pos > 0.55:
            steer -= 0.28
        elif track_pos > 0.35:
            steer -= 0.20
        elif track_pos > 0.20:
            steer -= 0.12

    # Soft return from edge.
    if track_pos < -0.75:
        steer += 0.18
    elif track_pos < -0.55:
        steer += 0.10

    if track_pos > 0.75:
        steer -= 0.18
    elif track_pos > 0.55:
        steer -= 0.10

    # Steering limits.
    if front < 25 and speed < 90:
        steer = clamp(steer, -0.62, 0.62)
    elif front < 45 and speed < 100:
        steer = clamp(steer, -0.55, 0.55)
    elif speed > 165:
        steer = clamp(steer, -0.42, 0.42)
    elif speed > 135:
        steer = clamp(steer, -0.50, 0.50)
    elif speed > 100:
        steer = clamp(steer, -0.60, 0.60)
    else:
        steer = clamp(steer, -0.70, 0.70)

    return steer


def calculate_legacy_target_speed(S):
    speed = S.get("speedX", 0.0)
    angle = abs(S.get("angle", 0.0))
    track_pos = S.get("trackPos", 0.0)
    track = S.get("track", [0.0] * 19)

    front = track[9]

    # Tight corners.
    if front < 8:
        return 64

    if front < 15:
        return 70

    if front < 25:
        return 81

    if front < 40:
        return 96

    if front < 60:
        return 110

    # Angle safety.
    if angle > 0.30 and speed > 90:
        return 72

    if angle > 0.24 and speed > 100:
        return 86

    if angle > 0.18 and speed > 120:
        return 108

    # Track edge safety.
    if abs(track_pos) > 0.88:
        return 68

    if abs(track_pos) > 0.72:
        return 88

    # Early slowdown before closing corners.
    if speed > 170 and front < 190 and FRONT_TREND < -1.0:
        return 170

    if speed > 155 and front < 160 and FRONT_TREND < -1.2:
        return 150

    if speed > 135 and front < 130 and FRONT_TREND < -1.5:
        return 128

    if speed > 115 and front < 100 and FRONT_TREND < -1.5:
        return 108

    # Medium distances.
    if front < 85:
        return 118

    if front < 120:
        return 150

    if front < 160:
        return 178

    return TARGET_SPEED



def is_final_turn_zone_dist(dist):
    """Last-corner entry/turn-in zone where high blend caused under-braking."""
    if not FINAL_TURN_GUARD_ENABLED:
        return False
    dist = float(dist) % TRACK_LENGTH_M
    return FINAL_TURN_START_M <= dist <= FINAL_TURN_END_M


def get_local_track_plan_blend(S):
    """
    Use aggressive global blend everywhere, but cap it in the final corner.
    This keeps the 1:49 pace without repeating the blend70 gravel exit.
    """
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    if is_final_turn_zone_dist(dist):
        return min(TRACK_PLAN_BLEND, FINAL_TURN_MAX_BLEND)
    return TRACK_PLAN_BLEND


def apply_final_turn_target_guard(S, target):
    """
    V7 hard local target-speed cap for the last corner only.
    The V6 guard was too soft: blend68 still under-braked and arrived too hot.
    This cap starts earlier and forces a lower entry target before turn-in.
    """
    if not FINAL_TURN_GUARD_ENABLED:
        return target

    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    track = S.get("track", [0.0] * 19)
    front = float(track[9])

    cap = None
    if 3050.0 <= dist < 3100.0:
        cap = 186.0
    elif 3100.0 <= dist < 3125.0:
        cap = 178.0
    elif 3125.0 <= dist < 3150.0:
        cap = 170.0
    elif 3150.0 <= dist < 3175.0:
        cap = 158.0
    elif 3175.0 <= dist < 3200.0:
        cap = 142.0
    elif 3200.0 <= dist < 3225.0:
        cap = 120.0
    elif 3225.0 <= dist < 3250.0:
        cap = 96.0
    elif 3250.0 <= dist < 3275.0:
        cap = 74.0
    elif 3275.0 <= dist < 3325.0:
        cap = 84.0

    if cap is not None and front < 22.0 and speed > 72.0:
        cap = min(cap, 72.0)
    if cap is not None and front < 14.0 and speed > 60.0:
        cap = min(cap, 66.0)

    if cap is None:
        return target
    return min(float(target), cap)

def calculate_final_turn_guard_brake(S):
    """
    V7 hard dedicated brake floor for the last corner.
    It starts before the previous guard and uses stronger brake floors.
    Still local only: global preview braking remains OFF.
    """
    if not FINAL_TURN_GUARD_ENABLED:
        return 0.0

    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9])

    # If already rotating hard/off-line, avoid adding more brake instability.
    if angle_abs > 0.36 or track_pos_abs > 0.82:
        return 0.0

    brake = 0.0

    if 3050.0 <= dist < 3100.0 and speed > 188.0:
        brake = max(brake, 0.05)
    if 3100.0 <= dist < 3125.0 and speed > 182.0:
        brake = max(brake, 0.08)
    if 3125.0 <= dist < 3150.0 and speed > 174.0:
        brake = max(brake, 0.12)
    if 3150.0 <= dist < 3175.0 and speed > 162.0:
        brake = max(brake, 0.16)
    if 3175.0 <= dist < 3200.0 and speed > 146.0:
        brake = max(brake, 0.22)
    if 3200.0 <= dist < 3225.0 and speed > 124.0:
        brake = max(brake, 0.30)
    if 3225.0 <= dist < 3250.0 and speed > 98.0:
        brake = max(brake, 0.34)
    if 3250.0 <= dist < 3275.0 and speed > 76.0 and front < 24.0:
        brake = max(brake, 0.26)

    # Extra emergency-ish local floor if sensor confirms the corner is closing.
    if 3150.0 <= dist < 3225.0 and front < 55.0 and speed > 120.0:
        brake = max(brake, 0.24)
    if 3225.0 <= dist < 3275.0 and front < 22.0 and speed > 78.0:
        brake = max(brake, 0.30)

    return clamp(brake, 0.0, 0.38)

def apply_final_turn_steer_assist(S, steer):
    """
    V7 earlier turn-in assist for the final corner.
    Brake is the main fix; this only prevents late rotation after the car is slowed.
    """
    if not FINAL_TURN_STEER_ASSIST_ENABLED:
        return steer

    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    track_pos = float(S.get("trackPos", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9])

    if angle_abs > 0.28:
        return steer

    boost = 0.0
    if 3175.0 <= dist < 3200.0 and speed > 135.0 and front < 90.0:
        boost = 0.012
    elif 3200.0 <= dist < 3225.0 and speed > 115.0 and front < 72.0:
        boost = 0.026
    elif 3225.0 <= dist < 3250.0 and speed > 88.0 and front < 46.0:
        boost = 0.046
    elif 3250.0 <= dist < 3275.0 and speed > 62.0 and front < 25.0:
        boost = 0.055

    # If still outside before turn-in, help a little more; avoid oversteer inside.
    if boost > 0.0 and track_pos > 0.02:
        boost += 0.010
    if track_pos < -0.35:
        boost *= 0.5

    return steer + boost


def apply_safe_straight_microbrake_cleaner(S, brake_value):
    """
    V12 safety-preserving top-speed helper.

    It removes only tiny leftover brake on very straight, very safe parts.
    It does NOT change real corner braking and does NOT imitate V11 late braking.
    """
    brake_value = float(brake_value)
    if brake_value <= 0.0:
        return 0.0

    speed = float(S.get("speedX", 0.0))
    angle = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z = float(S.get("speedZ", 0.0))
    track = S.get("track", [0.0] * 19)
    front = float(track[9])
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M

    # Never weaken the final corner guard or real corner safety.
    if is_final_turn_zone_dist(dist):
        return brake_value
    if front < 150.0:
        return brake_value
    if angle > 0.055:
        return brake_value
    if track_pos_abs > 0.38:
        return brake_value
    if abs(speed_z) > 2.0:
        return brake_value
    if FRONT_TREND < -0.65:
        return brake_value

    # Only clean tiny residual brake that blocks top speed on straights.
    if speed > 150.0 and brake_value <= 0.065:
        return 0.0

    return brake_value

def calculate_target_speed(S):
    """
    Stage-1 default: exact legacy target speed.

    Optional modes:
    - ENABLE_DYNAMIC_TARGET_SPEED: blend old target with adaptive formula.
    - ENABLE_SECTOR_SPEED_PROFILE: add per-sector speed bonus from sector_profile.json.
    """
    legacy_target = calculate_legacy_target_speed(S)
    target = legacy_target

    if ENABLE_DYNAMIC_TARGET_SPEED:
        dynamic_target = calculate_dynamic_profile(S)["dynamic_target_speed"]
        target = lerp(legacy_target, dynamic_target, DYNAMIC_TARGET_BLEND)

    if ENABLE_SECTOR_SPEED_PROFILE:
        profile = load_sector_speed_profile_once()
        sector_id = get_sector_id(S)
        sector_cfg = profile.get(sector_id, {})
        speed_bonus = float(sector_cfg.get("speed_bonus", 0.0))
        target += speed_bonus

    if ENABLE_TRACK_PLAN:
        planned_speed = get_track_planned_speed(S, 0.0)
        if planned_speed is not None:
            target = lerp(target, max(target, planned_speed), get_local_track_plan_blend(S))

    target = apply_final_turn_target_guard(S, target)

    return clamp(target, 55.0, TARGET_SPEED + 25.0)

def apply_brakes(S):
    speed = S.get("speedX", 0.0)
    angle = abs(S.get("angle", 0.0))
    track = S.get("track", [0.0] * 19)
    speed_z = S.get("speedZ", 0.0)
    track_pos = S.get("trackPos", 0.0)

    front = track[9]
    dist = S.get("distFromStart", 0.0)

    target_speed = calculate_target_speed(S)
    overspeed = speed - target_speed

    preview_brake = calculate_preview_brake(S)
    if preview_brake > 0.0:
        return apply_sector_brake_scale(S, preview_brake)

    final_turn_guard_brake = calculate_final_turn_guard_brake(S)
    if final_turn_guard_brake > 0.0:
        return final_turn_guard_brake

    # Tight corner brake.
    if front < 8:
        if speed > 68:
            return apply_sector_brake_scale(S, 0.20)
        if speed > 62 and abs(track_pos) > 0.62:
            return apply_sector_brake_scale(S, 0.12)
        return apply_sector_brake_scale(S, 0.0)
    if front < 15:
        if speed > 76:
            return apply_sector_brake_scale(S, 0.24)
        if speed > 68 and angle > 0.17:
            return apply_sector_brake_scale(S, 0.16)
        if speed > 64 and abs(track_pos) > 0.62:
            return apply_sector_brake_scale(S, 0.10)
        return apply_sector_brake_scale(S, 0.0)
    if front < 25:
        if speed > 88:
            return apply_sector_brake_scale(S, 0.28)
        if speed > 78 and angle > 0.15:
            return apply_sector_brake_scale(S, 0.18)
        if speed > 70 and abs(track_pos) > 0.58:
            return apply_sector_brake_scale(S, 0.10)
        return apply_sector_brake_scale(S, 0.0)
    # Angle safety.
    if angle > 0.30 and speed > 82:
        return apply_sector_brake_scale(S, 0.20)
    if angle > 0.24 and speed > 96:
        return apply_sector_brake_scale(S, 0.16)
    if angle > 0.17 and speed > 118:
        return apply_sector_brake_scale(S, 0.12)
    # Off track / edge safety.
    if front < 0 and speed > 35:
        return apply_sector_brake_scale(S, 0.65)
    if abs(track_pos) > 1.05 and speed > 45:
        return apply_sector_brake_scale(S, 0.40)
    if abs(track_pos) > 0.95 and speed > 78:
        return apply_sector_brake_scale(S, 0.26)
    if abs(track_pos) > 0.84 and speed > 92:
        return apply_sector_brake_scale(S, 0.18)
    # Early braking.
    if speed > 170 and front < 190 and FRONT_TREND < -1.0:
        return apply_sector_brake_scale(S, 0.12)
    if speed > 155 and front < 160 and FRONT_TREND < -1.2:
        return apply_sector_brake_scale(S, 0.17)
    if speed > 135 and front < 130 and FRONT_TREND < -1.5:
        return apply_sector_brake_scale(S, 0.22)
    if speed > 115 and front < 100 and FRONT_TREND < -1.5:
        return apply_sector_brake_scale(S, 0.27)
    # Local soft brake cap: 2550–2750.
    if 2550.0 <= dist <= 2750.0:
        if speed_z < -3.0 and front < 35 and speed > 78:
            return apply_sector_brake_scale(S, 0.34)
        if front < 35 and speed > 98:
            return apply_sector_brake_scale(S, 0.26)
        if front < 55 and speed > 118:
            return apply_sector_brake_scale(S, 0.20)
    # Chicane / sharp close.
    if speed_z < -3.0 and front < 35 and speed > 78:
        return apply_sector_brake_scale(S, 0.40)
    if front < 35 and speed > 98:
        return apply_sector_brake_scale(S, 0.30)
    if front < 55 and speed > 118:
        return apply_sector_brake_scale(S, 0.24)
    # Speed error brake.
    if overspeed <= 7:
        return apply_sector_brake_scale(S, 0.0)
    if overspeed > 45:
        return apply_sector_brake_scale(S, 0.35)
    if overspeed > 30:
        return apply_sector_brake_scale(S, 0.22)
    if overspeed > 18:
        return apply_sector_brake_scale(S, 0.11)
    if overspeed > 8:
        return apply_sector_brake_scale(S, 0.05)
    return apply_sector_brake_scale(S, 0.0)


def apply_v29_early_brake_prep(S, brake_value):
    """
    V29 local brake preparation.

    Purpose: V28 max-power made the car arrive hotter and you observed late braking.
    This does NOT enable global preview braking. It only adds brake floors in known
    turn-in zones when speed is too high for the upcoming corner.
    """
    if not V29_EARLY_BRAKE_ENABLED:
        return clamp(float(brake_value), 0.0, 1.0)

    brake_value = float(brake_value)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z = float(S.get("speedZ", 0.0))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if track_pos_abs > 0.88 or angle_abs > 0.42:
        return clamp(brake_value, 0.0, 1.0)

    floor = 0.0

    # First sector / fast kink prep.
    if 190.0 <= dist < 250.0 and speed > 180.0 and front < 190.0:
        floor = max(floor, 0.08)

    # 800m complex: earlier but not panic braking.
    if 760.0 <= dist < 810.0 and speed > 150.0:
        floor = max(floor, 0.08)
    if 810.0 <= dist < 850.0 and speed > 135.0 and front < 95.0:
        floor = max(floor, 0.12)

    # 1075m complex.
    if 1035.0 <= dist < 1085.0 and speed > 150.0:
        floor = max(floor, 0.08)
    if 1085.0 <= dist < 1125.0 and speed > 135.0 and front < 95.0:
        floor = max(floor, 0.12)

    # 1600m exit/turn support.
    if 1560.0 <= dist < 1625.0 and speed > 145.0 and front < 145.0:
        floor = max(floor, 0.07)

    if 2460.0 <= dist < 2550.0 and speed > 155.0 and front < 170.0:
        floor = max(floor, 0.08)
    if 2630.0 <= dist < 2725.0 and speed > 135.0 and front < 105.0:
        floor = max(floor, 0.10)
    if 2725.0 <= dist < 2800.0 and speed > 120.0 and front < 80.0:
        floor = max(floor, 0.12)

    if 3000.0 <= dist < 3050.0 and speed > 190.0 and front < 210.0:
        floor = max(floor, 0.07)
    if 3050.0 <= dist < 3125.0 and speed > 178.0:
        floor = max(floor, 0.10)
    if 3125.0 <= dist < 3200.0 and speed > 150.0:
        floor = max(floor, 0.16)
    if 3200.0 <= dist < 3275.0 and speed > 105.0:
        floor = max(floor, 0.22)

    # Downhill/closing-road support.
    if speed_z < -3.0 and front < 45.0 and speed > 90.0:
        floor = max(floor, 0.20)

    floor *= V29_EARLY_BRAKE_STRENGTH
    return clamp(max(brake_value, floor), 0.0, 0.42)


def apply_v34_sector77_edge_guard_brake(S, brake_value):
    """
    V34 local guard for the 1875-2000m complex.

    Fresh V29 logs showed that laps 1-5 survive this complex but already run
    very close to the edge in sectors 77/78. Lap 6 was only a few km/h hotter,
    crossed trackPos ~= -1 at ~1949m, then recovery made the situation worse.

    This function does not change the whole car. It adds a small brake floor
    only before and inside that high-risk complex, and only when the car is hot
    or already near the edge.
    """
    if not V34_SECTOR77_GUARD_ENABLED:
        return clamp(float(brake_value), 0.0, 1.0)

    brake_value = float(brake_value)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos = float(S.get("trackPos", 0.0))
    track_pos_abs = abs(track_pos)
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    floor = 0.0

    if 1760.0 <= dist < 1825.0 and speed > 222.0:
        floor = max(floor, V34_PRE_FLOOR_1)

    # Real braking corridor into the dangerous left-edge complex.
    if 1825.0 <= dist < 1875.0 and speed > 198.0:
        floor = max(floor, V34_PRE_FLOOR_2)

    if 1875.0 <= dist < 1925.0:
        if speed > 150.0:
            floor = max(floor, V34_PRE_FLOOR_3)
        if track_pos_abs > 0.72 and speed > 125.0:
            floor = max(floor, max(V34_PRE_FLOOR_3, 0.34 * V34_SECTOR77_BRAKE_STRENGTH))

    # The exact failure begins around 1925-1950: trackPos was already near -1.
    if 1925.0 <= dist < 1955.0:
        if speed > 98.0:
            floor = max(floor, 0.34 * V34_SECTOR77_BRAKE_STRENGTH)
        if track_pos_abs > 0.82 and speed > 78.0:
            floor = max(floor, V34_EDGE_BRAKE_HIGH)

    # Once it is near the edge, prioritize surviving the lap over acceleration.
    if 1955.0 <= dist < 2005.0:
        if track_pos_abs > 0.88 and speed > 70.0:
            floor = max(floor, V34_EDGE_BRAKE_MID)

    # If the car is straight and safely centered, do not add useless drag.
    if floor > 0.0:
        if track_pos_abs < 0.35 and angle_abs < 0.030 and front > 80.0 and speed < 170.0:
            floor *= 0.45
        return clamp(max(brake_value, floor), 0.0, 0.58)

    return clamp(brake_value, 0.0, 1.0)


def apply_v34_sector77_edge_guard_accel(S, accel_value, brake_value):
    """
    Keep throttle from pushing the car farther out in the same 1875-2000m danger zone.
    """
    if not V34_SECTOR77_GUARD_ENABLED:
        return clamp(float(accel_value), 0.0, 1.0)

    accel_value = float(accel_value)
    brake_value = float(brake_value)
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    speed = float(S.get("speedX", 0.0))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    angle_abs = abs(float(S.get("angle", 0.0)))

    if 1875.0 <= dist < 2005.0:
        if track_pos_abs > 0.92 and speed > 65.0:
            return 0.0
        if track_pos_abs > 0.82 and speed > 85.0:
            return min(accel_value, 0.25)
        if brake_value > 0.18:
            return min(accel_value, 0.35)
        if angle_abs > 0.35 and speed > 80.0:
            return min(accel_value, 0.20)

    return clamp(accel_value, 0.0, 1.0)


def calculate_throttle(S, R):
    front = S.get("track", [0.0] * 19)[9]
    angle = abs(S.get("angle", 0.0))
    speed_z = S.get("speedZ", 0.0)
    track_pos = S.get("trackPos", 0.0)
    speed = S.get("speedX", 0.0)

    target_speed = calculate_target_speed(S)
    speed_error = target_speed - speed

    # Off-track / edge / slide safety.
    if abs(track_pos) > 1.05:
        return apply_sector_accel_bias(S, 0.0)
    if abs(track_pos) > 0.95 and speed > 75:
        return apply_sector_accel_bias(S, 0.0)
    if angle > 0.85 and speed > 45:
        return apply_sector_accel_bias(S, 0.0)
    if front < 20 and speed > 55:
        return apply_sector_accel_bias(S, 0.0)
    if speed_z < -3.0 and front < 25 and speed > 55:
        return apply_sector_accel_bias(S, 0.0)
    # Chicane support throttle.
    if speed_z < -3.0 and front < 42 and speed > 55:
        if angle < 0.35 and abs(track_pos) < 0.75:
            return apply_sector_accel_bias(S, 0.35)
        return apply_sector_accel_bias(S, 0.0)
    # Straight.
    if (
        front > 130
        and angle < 0.22
        and abs(track_pos) < 0.80
        and speed < TARGET_SPEED - 5
    ):
        if speed_error > 20:
            return apply_sector_accel_bias(S, 1.0)
        if speed_error > 10:
            return apply_sector_accel_bias(S, 0.90)
        return apply_sector_accel_bias(S, 0.75)
    # Slow recovery / safe acceleration zone.
    if (
        front > 45
        and speed < 105
        and angle < 0.60
        and abs(track_pos) < 0.85
        and FRONT_TREND > -2.5
    ):
        if speed_error > 25:
            return apply_sector_accel_bias(S, 1.0)
        if speed_error > 15:
            return apply_sector_accel_bias(S, 0.90)
        if speed_error > 8:
            return apply_sector_accel_bias(S, 0.75)
        if speed_error > 3:
            return apply_sector_accel_bias(S, 0.55)
        return apply_sector_accel_bias(S, 0.30)
    # Corner exit.
    if (
        FRONT_TREND > 1.5
        and front > 45
        and speed < 130
        and angle < 0.55
        and abs(track_pos) < 0.85
    ):
        if speed_error > 25:
            return apply_sector_accel_bias(S, 1.0)
        if speed_error > 12:
            return apply_sector_accel_bias(S, 0.90)
        return apply_sector_accel_bias(S, 0.70)
    # Normal corner exit.
    if (
        speed < target_speed - 3
        and front > 45
        and angle < 0.60
        and abs(track_pos) < 0.90
    ):
        if speed_error > 25:
            return apply_sector_accel_bias(S, 1.0)
        if speed_error > 12:
            return apply_sector_accel_bias(S, 0.90)
        if speed_error > 5:
            return apply_sector_accel_bias(S, 0.70)
        return apply_sector_accel_bias(S, 0.45)
    # General speed controller.
    target_speed -= abs(R.get("steer", 0.0)) * 2.0
    speed_error = target_speed - speed

    if speed_error > 30:
        accel = 1.0
    elif speed_error > 18:
        accel = 0.90
    elif speed_error > 8:
        accel = 0.70
    elif speed_error > 2:
        accel = 0.50
    elif speed_error > -5:
        accel = 0.25
    else:
        accel = 0.0

    if speed < 20:
        accel = max(accel, 0.85)

    return apply_sector_accel_bias(S, clamp(accel, 0.0, 1.0))


def smooth_accel(target_accel):
    """V28: build throttle as fast as possible, but still keep a tiny numerical ramp."""
    global PREV_ACCEL

    target_accel = clamp(target_accel, 0.0, 1.0)

    if target_accel > PREV_ACCEL:
        press_speed = V28_ACCEL_PRESS_STEP
        accel = min(target_accel, PREV_ACCEL + press_speed)
    else:
        # Fast release is still needed when braking/saving the car.
        release_speed = 0.70
        accel = max(target_accel, PREV_ACCEL - release_speed)

    PREV_ACCEL = accel
    return accel


def smooth_brake(target_brake):
    global PREV_BRAKE

    target_brake = clamp(target_brake, 0.0, 1.0)

    if target_brake > PREV_BRAKE:
        if target_brake > 0.55:
            press_speed = 0.10
        elif target_brake > 0.30:
            press_speed = 0.08
        else:
            press_speed = 0.06

        brake = min(target_brake, PREV_BRAKE + press_speed)
    else:
        release_speed = 0.45
        brake = max(target_brake, PREV_BRAKE - release_speed)

    PREV_BRAKE = brake
    return brake


def v28_virtual_rpm(S):
    return float(S.get("rpm", 0.0)) * V28_RPM_VIRTUAL_SCALE


def v28_is_exit_or_straight(S):
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z = abs(float(S.get("speedZ", 0.0)))

    if track_pos_abs > 0.88 or speed_z > 4.2:
        return False
    if front > V28_STRAIGHT_FRONT_MIN and angle_abs < 0.13:
        return True
    if v23_in_any_range(dist, V23_EXIT_RANGES):
        return front > V28_EXIT_FRONT_MIN and angle_abs < V28_EXIT_MAX_ANGLE and track_pos_abs < 0.82
    return front > 70.0 and angle_abs < 0.20 and track_pos_abs < 0.78


def apply_v28_max_power(S, accel_value, brake_value):
    """
    V28 max acceleration layer.
    It forces full throttle only when the car is straight enough or in a known safe exit.
    It does not override real braking or off-track recovery.
    """
    accel_value = float(accel_value)
    brake_value = float(brake_value)
    speed = float(S.get("speedX", 0.0))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if brake_value > 0.16:
        return clamp(accel_value, 0.0, 1.0)
    if track_pos_abs > 0.90 or angle_abs > 0.55 or front < 22.0:
        return clamp(accel_value, 0.0, 1.0)

    # Clean acceleration zones: build speed aggressively up to the new target.
    if v28_is_exit_or_straight(S) and speed < TARGET_SPEED - 2.0:
        if brake_value < 0.075:
            return 1.0
        return max(accel_value, 0.88)

    # Slightly less clean but still safe: do not allow lazy 0.45-0.70 throttle.
    if front > 55.0 and angle_abs < 0.28 and track_pos_abs < 0.82 and speed < TARGET_SPEED - 12.0 and brake_value < 0.06:
        return max(accel_value, 0.92)

    return clamp(accel_value, 0.0, 1.0)


def shift_gears(S):
    """
    V29 real high-RPM / formula-style gearbox.

    Uses raw TORCS RPM directly. Upshift happens near the real tachometer top-end. The gearbox is deliberately simple:
    high-RPM upshift, limited exit downshift, cooldown to avoid hunting.
    """
    global V28_SHIFT_LOCK, V28_LAST_GEAR_CMD

    speed = float(S.get("speedX", 0.0))
    raw_rpm = float(S.get("rpm", 0.0))
    vrpm = raw_rpm * V28_RPM_VIRTUAL_SCALE
    gear = int(S.get("gear", 1) or 1)
    if gear < 1 or gear > 6:
        gear = 1

    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M

    if V28_SHIFT_LOCK > 0:
        V28_SHIFT_LOCK -= 1
        if 1 <= V28_LAST_GEAR_CMD <= 6:
            return int(V28_LAST_GEAR_CMD)
        return gear

    if speed < 8.0:
        V28_LAST_GEAR_CMD = 1
        return 1

    in_exit = v23_in_any_range(dist, V23_EXIT_RANGES)
    in_turn = v23_in_any_range(dist, V23_TURNIN_RANGES)
    clean_straight = front > 135.0 and angle_abs < 0.070 and track_pos_abs < 0.60
    mild_corner_or_exit = (front > 38.0 and angle_abs < 0.32 and track_pos_abs < 0.84) or in_exit

    # Hard sanity: if the car is in a gear that is clearly too high for the speed,
    # step down once. This is not a hunting gearbox; cooldown follows immediately.
    max_gear_by_speed = 1
    if speed > 34.0:
        max_gear_by_speed = 2
    if speed > 62.0:
        max_gear_by_speed = 3
    if speed > 98.0:
        max_gear_by_speed = 4
    if speed > 145.0:
        max_gear_by_speed = 5
    if speed > 195.0:
        max_gear_by_speed = 6

    if gear > max_gear_by_speed + 1:
        new_gear = max(1, gear - 1)
        V28_LAST_GEAR_CMD = new_gear
        V28_SHIFT_LOCK = V28_SHIFT_LOCK_STEPS
        return new_gear

    # Exit pull: one downshift only if virtual rpm is below powerband and the car is safe.
    if gear > 2 and in_exit and mild_corner_or_exit and vrpm < V28_DOWNSHIFT_VIRTUAL_RPM:
        max_downshift_speed = {3: 78.0, 4: 118.0, 5: 162.0, 6: 202.0}
        if speed < max_downshift_speed.get(gear, 999.0):
            new_gear = gear - 1
            V28_LAST_GEAR_CMD = new_gear
            V28_SHIFT_LOCK = V28_SHIFT_LOCK_STEPS + 2
            return new_gear

    # High-RPM upshift. In exits/corners it is allowed only when reasonably safe,
    # because the user specifically observed lost acceleration from not shifting there.
    min_speed_for_upshift = {1: 24.0, 2: 46.0, 3: 78.0, 4: 116.0, 5: 165.0}
    if gear < 6 and vrpm >= V28_UPSHIFT_VIRTUAL_RPM and speed > min_speed_for_upshift.get(gear, 0.0):
        if clean_straight or mild_corner_or_exit:
            new_gear = gear + 1
            V28_LAST_GEAR_CMD = new_gear
            V28_SHIFT_LOCK = V28_SHIFT_LOCK_STEPS
            return new_gear

    # Emergency anti-overrev: if raw rpm is already extremely high, shift even if
    # virtual settings were too conservative. This protects against never shifting.
    if gear < 6 and raw_rpm > max(19200.0, 19000.0) and speed > min_speed_for_upshift.get(gear, 0.0):
        new_gear = gear + 1
        V28_LAST_GEAR_CMD = new_gear
        V28_SHIFT_LOCK = V28_SHIFT_LOCK_STEPS
        return new_gear

    V28_LAST_GEAR_CMD = gear
    return gear


def traction_control(S, accel):
    if not ENABLE_TRACTION_CONTROL:
        return max(0.0, accel)

    wheel_spin = S.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])

    rear_slip = (wheel_spin[2] + wheel_spin[3]) - (wheel_spin[0] + wheel_spin[1])

    if rear_slip > 2:
        accel -= 0.1

    return max(0.0, accel)


# =============================================================================
# 7. NEURAL RUNTIME CONTROL / DEBUG
# =============================================================================

def debug_runtime_neural_only(S, R):
    """
    Runtime neural debug only.

    This function does NOT change:
    - steer
    - accel
    - brake
    - gear

    It only measures whether online TensorFlow prediction causes lag.
    """
    global BC_DEBUG_COUNTER

    BC_DEBUG_COUNTER += 1

    if BC_DEBUG_COUNTER % BC_DEBUG_EVERY_N_STEPS != 0:
        return

    speed = S.get("speedX", 0.0)
    track = S.get("track", [0.0] * 19)
    front = track[9]

    if speed < NEURAL_MIN_SPEED:
        return

    if front < NEURAL_MIN_FRONT:
        return

    t0 = time.perf_counter()
    pred = predict_bc_action_fast(S)
    t1 = time.perf_counter()

    predict_ms = (t1 - t0) * 1000.0
    mark_neural_called(predict_ms)

    if pred is None:
        return

    # Do not apply anything here.
    return


def apply_neural_steer_only(S, R):
    """
    Very safe neural steering assist.

    Rule-based steering remains primary.
    Neural model can only add a tiny correction.
    """
    speed = S.get("speedX", 0.0)
    angle = S.get("angle", 0.0)
    angle_abs = abs(angle)
    track_pos = S.get("trackPos", 0.0)
    speed_z = S.get("speedZ", 0.0)
    track = S.get("track", [0.0] * 19)

    front = track[9]
    dist_raced = S.get("distRaced", 0.0)

    rule_steer = float(R.get("steer", 0.0))

    # First full lap protection.
    if dist_raced < 3700.0:
        R["steer"] = rule_steer
        return R

    # Hard safety gates.
    if speed < NEURAL_MIN_SPEED:
        R["steer"] = rule_steer
        return R

    if front < NEURAL_MIN_FRONT:
        R["steer"] = rule_steer
        return R

    if front > NEURAL_MAX_FRONT:
        R["steer"] = rule_steer
        return R

    if angle_abs > NEURAL_MAX_ANGLE:
        R["steer"] = rule_steer
        return R

    if abs(track_pos) > NEURAL_MAX_TRACKPOS:
        R["steer"] = rule_steer
        return R

    if abs(speed_z) > NEURAL_MAX_SPEEDZ:
        R["steer"] = rule_steer
        return R

    # On near-straights, neural correction is blocked to avoid wobbling.
    if abs(rule_steer) < 0.025:
        R["steer"] = rule_steer
        return R

    t0 = time.perf_counter()
    pred = predict_bc_action_fast(S)
    t1 = time.perf_counter()

    predict_ms = (t1 - t0) * 1000.0
    mark_neural_called(predict_ms)

    if pred is None:
        R["steer"] = rule_steer
        return R

    neural_steer, neural_accel, neural_brake = pred
    neural_steer = float(np.clip(neural_steer, -0.75, 0.75))

    # Block opposite steering.
    opposite_sign = (
        abs(rule_steer) > 0.08
        and abs(neural_steer) > 0.08
        and rule_steer * neural_steer < 0.0
    )

    if opposite_sign:
        R["steer"] = rule_steer
        return R

    correction = neural_steer - rule_steer
    correction = float(np.clip(correction, -NEURAL_MAX_CORRECTION, NEURAL_MAX_CORRECTION))

    final_steer = rule_steer + NEURAL_STEER_BLEND * correction

    # Final protection: neural cannot sharply move steering.
    final_steer = float(np.clip(final_steer, rule_steer - 0.010, rule_steer + 0.010))
    final_steer = float(np.clip(final_steer, -0.75, 0.75))

    R["steer"] = final_steer
    mark_neural_applied()

    return R





V41_PROFILE_NAME = "carry"
V41_ENABLED = True
V41_PEAK_MIN_STEER = 0.145
V41_MAX_STEER_ADD_PER_STEP = 0.034
V41_STRAIGHT_ACCEL_CAP = 0.340
V41_COMMITTED_ACCEL_FLOOR = 0.900
V41_MICRO_BRAKE = 0.100

# (turn_start, apex_start, apex_end, exit_end, direction)
# Only the two slow hairpins reported by the driver. Direction follows the
V41_HAIRPIN_TURN_ZONES = [
    (405.0, 438.0, 485.0, 532.0, +1.0),
    (1488.0, 1534.0, 1588.0, 1642.0, +1.0),
]


def v41_progress(x, a, b):
    if b <= a:
        return 1.0
    return clamp((float(x) - float(a)) / (float(b) - float(a)), 0.0, 1.0)


def v41_hairpin_state(S):
    if not V41_ENABLED:
        return None
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    for turn_start, apex_start, apex_end, exit_end, direction in V41_HAIRPIN_TURN_ZONES:
        if turn_start <= dist < exit_end:
            if dist < apex_start:
                phase = "turnin"
                p = v41_progress(dist, turn_start, apex_start)
                req = V41_PEAK_MIN_STEER * (0.45 + 0.55 * p)
            elif dist < apex_end:
                phase = "apex"
                p = v41_progress(dist, apex_start, apex_end)
                req = V41_PEAK_MIN_STEER
            else:
                phase = "exit"
                p = 1.0 - v41_progress(dist, apex_end, exit_end)
                req = V41_PEAK_MIN_STEER * (0.38 + 0.62 * p)
            return {
                "dist": dist,
                "phase": phase,
                "direction": float(direction),
                "required_steer": float(req),
                "turn_start": turn_start,
                "apex_start": apex_start,
                "apex_end": apex_end,
                "exit_end": exit_end,
            }
    return None


def apply_v41_hairpin_turn_hold(S, steer):
    """Keep the car from accidentally straightening in the slow hairpins."""
    st = v41_hairpin_state(S)
    if st is None:
        return clamp(float(steer), -0.70, 0.70)

    steer = float(steer)
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    # Do not fight the car during slide/off-track situations. This is a hold,
    # not a forced snap-turn.
    if speed < 42.0 or speed > 138.0 or angle_abs > 0.42 or track_pos_abs > 0.98 or front < -0.1:
        return clamp(steer, -0.70, 0.70)

    direction = st["direction"]
    required = st["required_steer"]

    # If road is very closed, keep a bit more turn commitment. If front is open,
    # reduce the hold so the car can unwind naturally.
    if front < 10.0:
        required *= 1.10
    elif front > 35.0 and st["phase"] == "exit":
        required *= 0.72

    # If the car is already very close to track edge, avoid adding large steering
    # jumps. The normal safety/recovery logic will handle it.
    if track_pos_abs > 0.88:
        required *= 0.70

    current_signed = steer * direction
    if current_signed < required:
        add = min(required - current_signed, V41_MAX_STEER_ADD_PER_STEP)
        steer += direction * add

    return clamp(steer, -0.66, 0.66)


def apply_v41_hairpin_no_straight_throttle(S, steer, accel_value, brake_value):
    """Block the exact failure mode: steering drops, car accelerates straight."""
    st = v41_hairpin_state(S)
    if st is None:
        return clamp(float(accel_value), 0.0, 1.0), clamp(float(brake_value), 0.0, 1.0)

    steer = float(steer)
    accel_value = float(accel_value)
    brake_value = float(brake_value)
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z_abs = abs(float(S.get("speedZ", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    if speed < 45.0 or track_pos_abs > 1.02:
        return clamp(accel_value, 0.0, 1.0), clamp(brake_value, 0.0, 1.0)

    direction = st["direction"]
    required = st["required_steer"]
    signed_steer = steer * direction
    too_straight = signed_steer < required * 0.62
    still_closed = front < 26.0 or st["phase"] in ("turnin", "apex")

    # Main fix: if the car is inside the hairpin but the steering commitment
    # vanished, do not let max-power/RPM layers accelerate it straight.
    if too_straight and still_closed and speed > 58.0:
        accel_value = min(accel_value, V41_STRAIGHT_ACCEL_CAP)
        if front < 13.0 and speed > 76.0 and angle_abs < 0.30:
            brake_value = max(brake_value, V41_MICRO_BRAKE)
        return clamp(accel_value, 0.0, 1.0), clamp(brake_value, 0.0, 0.58)

    committed = signed_steer >= required * 0.82
    safe_line = track_pos_abs < 0.84 and angle_abs < 0.36 and speed_z_abs < 5.8 and front > 7.0
    if committed and safe_line and brake_value < 0.16:
        accel_value = max(accel_value, V41_COMMITTED_ACCEL_FLOOR)
        if st["phase"] == "exit" and track_pos_abs < 0.76 and angle_abs < 0.30 and front > 11.0:
            accel_value = max(accel_value, min(1.0, V41_COMMITTED_ACCEL_FLOOR + 0.06))

    return clamp(accel_value, 0.0, 1.0), clamp(brake_value, 0.0, 0.58)



V42_PROFILE_NAME = "carry"
V42_ENABLED = True
V42_BRAKE_GAIN = 0.7
V42_ACCEL_CAP_PRE = 0.38
V42_ACCEL_CAP_MID = 0.5
V42_EXIT_ACCEL_FLOOR = 0.92
V42_TRACK_LIMIT = 0.88
V42_STEER_MIN_FACTOR = 0.54

# (guard_start, turn_start, apex_start, apex_end, exit_end, direction,
#  speed_pre, speed_apex, speed_exit)
# First slow hairpin and the second slow hairpin/slow loaded corner.
V42_HAIRPIN_GUARD_ZONES = [
    (382.0, 405.0, 438.0, 488.0, 535.0, +1.0, 108.0, 93.0, 106.0),
    (1450.0, 1488.0, 1534.0, 1590.0, 1645.0, +1.0, 121.0, 101.0, 115.0),
]


def v42_zone_state(S):
    if not V42_ENABLED:
        return None
    dist = float(S.get("distFromStart", 0.0)) % TRACK_LENGTH_M
    for guard_start, turn_start, apex_start, apex_end, exit_end, direction, sp_pre, sp_apex, sp_exit in V42_HAIRPIN_GUARD_ZONES:
        if guard_start <= dist < exit_end:
            if dist < turn_start:
                phase = "guard"
                p = v41_progress(dist, guard_start, turn_start)
                speed_limit = sp_pre + 4.0  # only trim extreme overspeed before real turn-in
            elif dist < apex_start:
                phase = "turnin"
                p = v41_progress(dist, turn_start, apex_start)
                speed_limit = sp_pre * (1.0 - 0.10 * p) + sp_apex * (0.10 * p)
            elif dist < apex_end:
                phase = "apex"
                p = v41_progress(dist, apex_start, apex_end)
                speed_limit = sp_apex
            else:
                phase = "exit"
                p = v41_progress(dist, apex_end, exit_end)
                speed_limit = sp_apex * (1.0 - p) + sp_exit * p
            return {
                "dist": dist,
                "phase": phase,
                "p": float(p),
                "direction": float(direction),
                "speed_limit": float(speed_limit),
                "guard_start": guard_start,
                "turn_start": turn_start,
                "apex_start": apex_start,
                "apex_end": apex_end,
                "exit_end": exit_end,
            }
    return None


def apply_v42_hairpin_speed_guard_brake(S, brake_value):
    """Short, late-ish brake guard: prevent overspeed, not a long early brake."""
    st = v42_zone_state(S)
    if st is None:
        return clamp(float(brake_value), 0.0, 1.0)

    brake_value = float(brake_value)
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    overspeed = speed - st["speed_limit"]
    if overspeed <= 0.0:
        return clamp(brake_value, 0.0, 0.62)

    # Before the real turn-in, only trim if the car is clearly too hot.
    if st["phase"] == "guard" and overspeed < 5.0:
        return clamp(brake_value, 0.0, 0.62)

    # If the car is already close to track limit or the road is closed, it needs
    # a sharper but still short brake pulse. This avoids the 60 km/h recovery.
    risk = 0.0
    if track_pos_abs > 0.78:
        risk += 0.05
    if track_pos_abs > V42_TRACK_LIMIT:
        risk += 0.08
    if front < 15.0:
        risk += 0.04
    if angle_abs < 0.20 and st["phase"] in ("turnin", "apex"):
        risk += 0.035

    if st["phase"] in ("turnin", "apex"):
        floor = 0.070 + V42_BRAKE_GAIN * min(0.24, overspeed * 0.018) + risk
    elif st["phase"] == "guard":
        floor = 0.045 + V42_BRAKE_GAIN * min(0.16, overspeed * 0.012) + risk * 0.60
    else:
        # Exit should not keep braking unless it is truly leaving the track.
        floor = 0.030 + V42_BRAKE_GAIN * min(0.10, overspeed * 0.010) + risk * 0.50

    return clamp(max(brake_value, floor), 0.0, 0.62)


def apply_v42_hairpin_overspeed_accel(S, steer, accel_value, brake_value):
    """Cut only the bad straight-line acceleration; let it carry once committed."""
    st = v42_zone_state(S)
    if st is None:
        return clamp(float(accel_value), 0.0, 1.0), clamp(float(brake_value), 0.0, 1.0)

    steer = float(steer)
    accel_value = float(accel_value)
    brake_value = float(brake_value)
    speed = float(S.get("speedX", 0.0))
    angle_abs = abs(float(S.get("angle", 0.0)))
    track_pos_abs = abs(float(S.get("trackPos", 0.0)))
    speed_z_abs = abs(float(S.get("speedZ", 0.0)))
    track = S.get("track", [0.0] * 19)
    front = float(track[9]) if len(track) > 9 else 0.0

    overspeed = speed - st["speed_limit"]
    signed_steer = steer * st["direction"]
    min_commit = V41_PEAK_MIN_STEER * V42_STEER_MIN_FACTOR
    too_straight = signed_steer < min_commit
    edge_risk = track_pos_abs > V42_TRACK_LIMIT or speed_z_abs > 6.2

    # If it is going into the hairpin too fast, do not feed power even when the
    # power/RPM layers request it. This is the exact track-limit failure mode.
    if st["phase"] in ("guard", "turnin", "apex") and (overspeed > 1.5 or too_straight or edge_risk):
        cap = V42_ACCEL_CAP_PRE if st["phase"] != "apex" else V42_ACCEL_CAP_MID
        if edge_risk:
            cap = min(cap, 0.18)
        if too_straight and speed > 62.0:
            cap = min(cap, 0.22)
        accel_value = min(accel_value, cap)

    # Once it is committed and not near the outside edge, do not be afraid of gas.
    committed = signed_steer >= min_commit * 1.15
    stable_line = track_pos_abs < 0.76 and angle_abs < 0.34 and speed_z_abs < 5.2 and front > 7.0
    if st["phase"] in ("apex", "exit") and committed and stable_line and brake_value < 0.14:
        accel_value = max(accel_value, V42_EXIT_ACCEL_FLOOR)
        if st["phase"] == "exit" and track_pos_abs < 0.68 and front > 11.0:
            accel_value = max(accel_value, min(1.0, V42_EXIT_ACCEL_FLOOR + 0.06))

    return clamp(accel_value, 0.0, 1.0), clamp(brake_value, 0.0, 0.62)


# =============================================================================
# 8. MAIN DRIVE FUNCTION
# =============================================================================

def drive_modular(c):
    S, R = c.S.d, c.R.d

    reset_neural_runtime_marks()
    update_front_trend(S)

    track = S.get("track", [0.0] * 19)
    front = track[9]
    angle = S.get("angle", 0.0)
    angle_abs = abs(angle)
    speed = S.get("speedX", 0.0)
    track_pos = S.get("trackPos", 0.0)

    # -------------------------------------------------------------------------
    # Recovery: car is already spun / reversing / sensor invalid.
    # -------------------------------------------------------------------------
    if speed < -2 or angle_abs > 2.2 or front < -0.5:
        if front < -0.5 and speed > 38.0:
            R["accel"] = 0.0
            R["brake"] = 0.34
        else:
            R["brake"] = 0.0
            R["accel"] = 0.25

        R["gear"] = 1

        if abs(track_pos) > 0.85:
            if track_pos > 0:
                R["steer"] = -0.35
            else:
                R["steer"] = 0.35
        else:
            if angle > 0:
                R["steer"] = -0.35
            else:
                R["steer"] = 0.35

        finish_frame_logging(S, R, target_brake=R.get("brake", 0.0))
        return

    # -------------------------------------------------------------------------
    # Steering: stable rule-based steering first.
    # -------------------------------------------------------------------------
    target_steer = calculate_steering(S)
    R["steer"] = smooth_steer(target_steer, S)
    R["steer"] = apply_v41_hairpin_turn_hold(S, R["steer"])

    # Neural is never called in pure rule-based mode.
    if USE_NEURAL_STEER and not ENABLE_RULE_BASED_ONLY:
        if USE_BC_DEBUG_ONLY or NEURAL_DRY_RUN or DEBUG_NEURAL_STEER:
            debug_runtime_neural_only(S, R)
        elif USE_SAFE_STEER_CORRECTION:
            R = apply_neural_steer_only(S, R)

    # -------------------------------------------------------------------------
    # Brake.
    # -------------------------------------------------------------------------
    target_brake = apply_brakes(S)
    target_brake = apply_v29_early_brake_prep(S, target_brake)
    target_brake = apply_v34_sector77_edge_guard_brake(S, target_brake)
    target_brake = apply_v35_centered_corner_brake(S, target_brake)
    target_brake = apply_v36_edge_apex_brake_phase(S, target_brake)
    R["brake"] = smooth_brake(target_brake)
    R["brake"] = apply_sector_final_brake_cap(S, R["brake"])
    R["brake"] = apply_safe_straight_microbrake_cleaner(S, R["brake"])
    R["brake"] = apply_v42_hairpin_speed_guard_brake(S, R["brake"])

    # -------------------------------------------------------------------------
    # Accel.
    # -------------------------------------------------------------------------
    target_accel = calculate_throttle(S, R)

    # Strong brake closes throttle.
    # Small brake does not kill acceleration completely.
    if R["brake"] > V39_BRAKE_KILL_ACCEL_THRESHOLD:
        target_accel = 0.0

    R["accel"] = smooth_accel(target_accel)
    R["accel"] = apply_sector_final_accel_floor(S, R["accel"], R["brake"])
    R["accel"] = apply_v23_power_commit(S, R["accel"], R["brake"])
    R["accel"] = apply_v28_max_power(S, R["accel"], R["brake"])
    R["accel"] = apply_v34_sector77_edge_guard_accel(S, R["accel"], R["brake"])
    R["accel"] = apply_v35_midcorner_gas_commit(S, R["accel"], R["brake"])
    R["accel"] = apply_v36_edge_apex_speed_commit(S, R["accel"], R["brake"])
    R["accel"], R["brake"] = apply_v41_hairpin_no_straight_throttle(S, R["steer"], R["accel"], R["brake"])
    R["accel"], R["brake"] = apply_v42_hairpin_overspeed_accel(S, R["steer"], R["accel"], R["brake"])

    # -------------------------------------------------------------------------
    # Safety: car is close to edge / leaving track.
    # -------------------------------------------------------------------------
    if abs(track_pos) > 0.95 and speed > 55:
        R["accel"] = 0.0
        R["brake"] = max(R["brake"], 0.15)

        # Steer back to center.
        if track_pos > 0:
            R["steer"] = min(R["steer"], -0.18)
        else:
            R["steer"] = max(R["steer"], 0.18)

        R["gear"] = shift_gears(S)

        finish_frame_logging(S, R, target_brake)
        return

    # -------------------------------------------------------------------------
    # Anti-slide: counter-steer if angle is too large.
    # -------------------------------------------------------------------------
    if angle_abs > 0.75 and speed > 25:
        R["accel"] = 0.0
        R["brake"] = 0.15

        if angle > 0:
            R["steer"] = -0.25
        else:
            R["steer"] = 0.25

        R["gear"] = shift_gears(S)

        finish_frame_logging(S, R, target_brake)
        return

    # -------------------------------------------------------------------------
    # Low-speed recovery / normal drive.
    # -------------------------------------------------------------------------
    if speed < 15 and front > 10 and angle_abs < 0.8:
        R["brake"] = 0.0
        R["accel"] = 0.8
        R["gear"] = 1
    else:
        if R["brake"] > 0.20:
            R["accel"] = 0.0
        else:
            R["accel"] = traction_control(S, R["accel"])
            R["accel"] = apply_sector_final_accel_floor(S, R["accel"], R["brake"])
            R["accel"] = apply_v23_power_commit(S, R["accel"], R["brake"])
            R["accel"] = apply_v28_max_power(S, R["accel"], R["brake"])

        R["accel"], R["brake"] = apply_v41_hairpin_no_straight_throttle(S, R["steer"], R["accel"], R["brake"])
        R["accel"], R["brake"] = apply_v42_hairpin_overspeed_accel(S, R["steer"], R["accel"], R["brake"])
        R["gear"] = shift_gears(S)

    # -------------------------------------------------------------------------
    # Logging and audit.
    # -------------------------------------------------------------------------
    finish_frame_logging(S, R, target_brake)

    # -------------------------------------------------------------------------
    # Optional print debug.
    # -------------------------------------------------------------------------
    if PRINT_DEBUG:
        wheel_spin = S.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])
        wheel_slip = (wheel_spin[2] + wheel_spin[3]) - (wheel_spin[0] + wheel_spin[1])
        current_target_speed = calculate_target_speed(S)

        print(
            "mode =", LAST_CONTROL_MODE,
            "dt_ms =", round(LAST_CONTROL_TIME or 0.0, 4),
            "predict_ms =", round(LAST_PREDICT_MS, 2),
            "target_speed =", round(current_target_speed, 1),
            "target_brake =", round(target_brake, 2),
            "real_brake =", round(R["brake"], 2),
            "accel =", round(R["accel"], 2),
            "gear =", R["gear"],
            "speed =", round(speed, 1),
            "front =", round(front, 1),
            "steer =", round(R["steer"], 2),
            "angle =", round(angle, 2),
            "speedZ =", round(S.get("speedZ", 0.0), 2),
            "trackPos =", round(track_pos, 2),
            "slip =", round(wheel_slip, 1),
            "rpm =", round(S.get("rpm", 0.0), 0),
        )

    return


# =============================================================================
# 9. MAIN LOOP
# =============================================================================

if __name__ == "__main__":
    C = Client(p=3001)

    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_modular(C)
        C.respond_to_server()

    C.shutdown()
