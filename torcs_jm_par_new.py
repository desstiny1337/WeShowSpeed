
import socket
import sys
import getopt
import os
import time
import csv
import math
import pandas as pd
import joblib
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



# =====================================================================
# ADAPTIVE STABLE DRIVER
# =====================================================================
# Цель этого файла:
#   сначала НЕ скорость, а стабильное прохождение круга без вылета.
#
# Что здесь убрано:
#   - ML модель;
#   - policy_model.pkl;
#   - pandas/joblib;
#   - left_open/right_open как основной ориентир;
#   - агрессивные фиксированные speed table.
#
# Что используется:
#   - front = track[9];
#   - front_delta;
#   - trackPos;
#   - angle;
#   - speedY / speedZ;
#   - общий risk;
#   - адаптивная aggression.
#
# Машина каждый шаг оценивает риск и сама снижает скорость,
# если видит закрытый поворот, край трассы, занос или боковое движение.
# =====================================================================

import math
import csv


# =====================================================================
# GLOBAL ADAPTIVE STATE
# =====================================================================

ADAPT = {
    # V6:
    # возвращаем разгон и потолок ближе к v3,
    # но оставляем локальные страховки шиканы и последнего поворота.
    "aggression": 0.90,
    "max_speed": 218.0,

    # Руль не должен держать машину намертво в центре,
    # но в опасных местах усилится через update_adaptive_parameters().
    "steer_gain": 1.35,
    "center_gain": 0.58,
    "brake_gain": 0.85,
}

PREV_FRONT = None
PREV_TRACK_POS = None
PREV_STEER = 0.0
PREV_ACCEL = 0.0
PREV_BRAKE = 0.0

ADAPT_STEP = 0

LOG_FILE = "adaptive_run_log.csv"
LOG_INITIALIZED = False

# Память по участкам трассы.
# Если сектор был опасным, скорость на нём ограничивается.
SECTOR_SIZE = 10.0
SECTOR_SPEED_CAP = {}


# =====================================================================
# BASIC HELPERS
# =====================================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def get_track(S):
    track = S.get("track", [200.0] * 19)

    if len(track) < 19:
        track = list(track) + [200.0] * (19 - len(track))

    return track


def get_sector_id(S):
    dist = S.get("distFromStart", 0.0)

    if dist < 0:
        dist = S.get("distRaced", 0.0)

    return int(dist // SECTOR_SIZE)


# =====================================================================
# STATE ANALYSIS
# =====================================================================

def analyze_state(S):
    """
    Собирает текущее состояние машины и считает риски.

    Важное исправление v5:
    раньше slide_risk считал speedZ как занос. В TORCS speedZ — это в основном
    вертикальное движение/неровности, а не боковой занос. Из-за этого на прямой
    перед шиканой risk становился почти 1.0, машина закрывала газ и ехала
    около 134 вместо дальнейшего разгона.

    Теперь slide_risk основан в основном на speedY.
    """

    global PREV_FRONT, PREV_TRACK_POS

    track = get_track(S)

    speed = S.get("speedX", 0.0)
    speed_y_raw = S.get("speedY", 0.0)
    speed_y = abs(speed_y_raw)
    speed_z = abs(S.get("speedZ", 0.0))

    angle = S.get("angle", 0.0)
    track_pos = S.get("trackPos", 0.0)

    front = track[9]

    if PREV_FRONT is None:
        front_delta = 0.0
    else:
        front_delta = front - PREV_FRONT

    PREV_FRONT = front

    if PREV_TRACK_POS is None:
        track_pos_delta = 0.0
    else:
        track_pos_delta = track_pos - PREV_TRACK_POS

    PREV_TRACK_POS = track_pos

    # ---------------------------------------------------------
    # Risk 1: впереди мало места.
    # ---------------------------------------------------------
    if front < 0:
        front_risk = 1.0
    else:
        front_risk = clamp((75.0 - front) / 75.0, 0.0, 1.0)

    # ---------------------------------------------------------
    # Risk 2: поворот быстро закрывается.
    # ---------------------------------------------------------
    closing_risk = clamp((-front_delta - 6.0) / 26.0, 0.0, 1.0)

    # ---------------------------------------------------------
    # Risk 3: близко к краю.
    # Не паникуем при 0.4-0.5, но жёстко реагируем ближе к 0.85.
    # ---------------------------------------------------------
    edge_risk = clamp((abs(track_pos) - 0.50) / 0.50, 0.0, 1.0)

    # ---------------------------------------------------------
    # Risk 4: угол машины.
    # ---------------------------------------------------------
    angle_risk = clamp((abs(angle) - 0.13) / 0.55, 0.0, 1.0)

    # ---------------------------------------------------------
    # Risk 5: боковой занос.
    # speedZ больше НЕ является основным источником риска.
    # ---------------------------------------------------------
    slide_risk = clamp((speed_y - 6.5) / 18.0, 0.0, 1.0)

    # Если одновременно большой угол и боковая скорость, усиливаем риск.
    if speed_y > 8.0 and abs(angle) > 0.25:
        slide_risk = max(slide_risk, 0.75)

    total_risk = max(
        front_risk,
        closing_risk,
        edge_risk,
        angle_risk,
        slide_risk,
    )

    return {
        "track": track,

        "speed": speed,
        "speed_y": speed_y,
        "speed_z": speed_z,

        "angle": angle,
        "track_pos": track_pos,
        "track_pos_delta": track_pos_delta,

        "front": front,
        "front_delta": front_delta,

        "front_risk": front_risk,
        "closing_risk": closing_risk,
        "edge_risk": edge_risk,
        "angle_risk": angle_risk,
        "slide_risk": slide_risk,
        "total_risk": total_risk,
    }

# =====================================================================
# ADAPTIVE PARAMETERS
# =====================================================================


def update_adaptive_parameters(S, state):
    """
    V6 adaptive parameters.

    Главная цель:
    - вернуть разгон и максималку ближе к v3;
    - не душить aggression из-за обычного риска;
    - не превращать sector_cap в ограничитель скорости на прямых.
    """

    sector = get_sector_id(S)

    risk = state["total_risk"]
    speed = state["speed"]
    front = state["front"]
    angle = abs(state["angle"])
    track_pos = abs(state["track_pos"])
    speed_y = state["speed_y"]

    dangerous = (
        abs(track_pos) > 0.92
        or angle > 0.52
        or speed_y > 13.0
        or front < 7
    )

    safe = (
        risk < 0.32
        and abs(track_pos) < 0.45
        and angle < 0.14
        and front > 55
        and speed > 20
    )

    if dangerous:
        ADAPT["aggression"] -= 0.003
    elif safe:
        ADAPT["aggression"] += 0.0032

    # Было максимум около 0.92 при max_speed=160 -> потолок 147.
    # Теперь даём машине выйти выше 150-160 на прямых.
    ADAPT["aggression"] = clamp(ADAPT["aggression"], 0.74, 1.02)

    # Меньше центрируем на спокойных участках, больше — только при риске.
    ADAPT["center_gain"] = 0.48 + 0.52 * risk
    ADAPT["brake_gain"] = 0.70 + 0.62 * risk
    ADAPT["steer_gain"] = 1.18 + 0.50 * risk

    # Sector memory оставляем мягкой. Она нужна только если реально опасно.
    if dangerous and speed > 45:
        for back in range(0, 2):
            s = sector - back
            old_cap = SECTOR_SPEED_CAP.get(s, ADAPT["max_speed"])
            new_cap = max(105.0, old_cap - 0.35)
            SECTOR_SPEED_CAP[s] = new_cap

    elif safe:
        old_cap = SECTOR_SPEED_CAP.get(sector, ADAPT["max_speed"])
        new_cap = min(ADAPT["max_speed"], old_cap + 1.3)
        SECTOR_SPEED_CAP[sector] = new_cap

# =====================================================================
# LOCAL TRACK-SPECIFIC HELPERS
# =====================================================================


CHICANE_PROFILE = [
    # V26: chicane restored exactly to the successful v19 profile.
    # No S-line, no aggressive feedforward steer.
    # dist, target_speed, target_trackPos, feedforward_steer
    (2180.0, 145.0,  0.00, 0.00),
    (2220.0, 128.0,  0.00, 0.00),
    (2260.0, 112.0,  0.00, 0.00),
    (2300.0,  92.0,  0.00, 0.00),
    (2325.0,  78.0,  0.00, 0.00),
    (2350.0,  64.0,  0.00, 0.00),
    (2380.0,  56.0,  0.00, 0.00),
    (2420.0,  54.0,  0.00, 0.00),
    (2460.0,  56.0,  0.00, 0.00),
    (2500.0,  64.0,  0.00, 0.00),
    (2530.0,  82.0,  0.00, 0.00),
    (2550.0, 108.0,  0.00, 0.00),
]

def interp_profile(dist, idx):
    points = CHICANE_PROFILE

    if dist <= points[0][0]:
        return points[0][idx]

    for i in range(len(points) - 1):
        d0 = points[i][0]
        d1 = points[i + 1][0]
        if d0 <= dist <= d1:
            t = (dist - d0) / max(1.0, d1 - d0)
            return points[i][idx] + (points[i + 1][idx] - points[i][idx]) * t

    return points[-1][idx]


def is_chicane_zone(S):
    """
    V19: chicane mode only near the real problematic turns.
    Before 2180 m we do not touch steering. From 2180 m we only slow down.
    """
    d = S.get("distFromStart", 0.0)
    return 2180.0 <= d <= 2550.0


def chicane_line_target(dist):
    """
    V16: целевая линия взята из нормального baseline-прохождения.

    Это не жёсткий центр. Шикана требует небольшого смещения:
    сначала около центра, потом коротко левее, затем выход правее/к центру.
    """
    return interp_profile(dist, 2)


def chicane_profile_speed(dist):
    return interp_profile(dist, 1)


def chicane_profile_steer(dist):
    return interp_profile(dist, 3)


def is_finish_corner_zone(S):
    """
    V21: финальный сектор больше НЕ является отдельным режимом.

    Просьба: сделать этот отрезок ничем не отличающимся от остальных
    поворотов трассы. Поэтому вся логика finish-corridor / anti-S выключена:
    участок проходит через общий adaptive_target_speed + adaptive_steer.
    """
    return False

def finish_line_target(dist):
    """
    V20: baseline-like линия финального сектора.

    В v19 шикана стала отличной, но после неё машина перед финальным
    поворотом начинала S-раскачку. Причина: старый finish target тянул
    машину к центру слишком рано.

    Поэтому 2860-3240 м держим стабильную правую/внешнюю полосу
    примерно +0.16 -> +0.06, как в хорошем baseline. Левый поворот
    начинается только после ~3245 м.
    """
    points = [
        (2860.0,  0.16),
        (2920.0,  0.17),
        (3000.0,  0.18),
        (3080.0,  0.15),
        (3160.0,  0.09),
        (3220.0,  0.06),
        (3245.0,  0.04),
        (3260.0, -0.18),
        (3275.0, -0.40),
        (3295.0, -0.48),
        (3335.0, -0.38),
        (3360.0, -0.30),
    ]

    if dist <= points[0][0]:
        return points[0][1]

    for i in range(len(points) - 1):
        d0, p0 = points[i]
        d1, p1 = points[i + 1]
        if d0 <= dist <= d1:
            t = (dist - d0) / max(1.0, d1 - d0)
            return p0 + (p1 - p0) * t

    return points[-1][1]


def chicane_speed_cap(S, state, target_speed):
    """
    V19: slow the chicane like an ordinary safe corner.

    The previous versions entered the first chicane turn too fast and then
    special profile steering pushed the car into the wall. Now speed is the
    main stabilizer: early braking, low mid-chicane speed, smooth exit.
    """
    d = S.get("distFromStart", 0.0)

    # Approach: only speed cap, no profile steering before the actual corner.
    if 2100.0 <= d < 2180.0:
        return min(target_speed, 158.0)
    if 2180.0 <= d < 2220.0:
        return min(target_speed, 142.0)
    if 2220.0 <= d < 2260.0:
        return min(target_speed, 124.0)
    if 2260.0 <= d < 2300.0:
        return min(target_speed, 106.0)

    if not (2300.0 <= d <= 2550.0):
        return target_speed

    cap = chicane_profile_speed(d)

    tp = abs(state["track_pos"])
    angle = abs(state["angle"])
    signed_speed_y = abs(S.get("speedY", 0.0))

    # If the car is leaving the road, slow more. Do not wait until trackPos=1.
    if tp > 0.42:
        cap = min(cap, 58.0)
    if tp > 0.58:
        cap = min(cap, 48.0)
    if tp > 0.74:
        cap = min(cap, 38.0)

    if angle > 0.34:
        cap = min(cap, 58.0)
    if angle > 0.50:
        cap = min(cap, 44.0)

    if signed_speed_y > 7.0:
        cap = min(cap, 56.0)
    if signed_speed_y > 10.0:
        cap = min(cap, 44.0)

    return min(target_speed, cap)

def finish_corner_speed_cap(S, state, target_speed):
    """
    V27: final braking by the visible 3-2-1 boards.

    Keep the successful chicane unchanged. The final sector uses only speed:
    fast until board 3, then 3 -> 2 -> 1 braking, and immediate release after
    the apex when the car is already pointed back down the straight.
    """
    d = S.get("distFromStart", 0.0)

    # Before the braking boards and after the apex we do not limit speed here.
    if not (3150.0 <= d <= 3294.0):
        return target_speed

    # Board 3 -> Board 2 -> Board 1 -> apex.
    # These caps are deliberately low near board 1, but the restriction ends
    # earlier than in v26 so acceleration after the turn is immediate.
    if d < 3180.0:        # board 3: prepare, do not kill speed
        cap = 142.0
    elif d < 3210.0:      # between 3 and 2
        cap = 108.0
    elif d < 3240.0:      # board 2
        cap = 74.0
    elif d < 3278.0:      # board 1 / turn-in
        cap = 48.0
    else:                 # apex; release after 3294
        cap = 46.0

    tp = abs(state["track_pos"])
    angle = abs(state["angle"])
    signed_speed_y = abs(S.get("speedY", 0.0))

    # If the car is not cleanly inside the turn, keep it slower.
    if tp > 0.45:
        cap = min(cap, 48.0)
    if tp > 0.65:
        cap = min(cap, 38.0)
    if angle > 0.36:
        cap = min(cap, 48.0)
    if angle > 0.55:
        cap = min(cap, 38.0)
    if signed_speed_y > 7.0:
        cap = min(cap, 46.0)
    if signed_speed_y > 10.0:
        cap = min(cap, 36.0)

    return min(target_speed, cap)

def local_speed_cap(S, state, target_speed):
    target_speed = chicane_speed_cap(S, state, target_speed)
    target_speed = finish_corner_speed_cap(S, state, target_speed)
    return target_speed




def is_post_chicane_straight(S):
    """
    V20: зона разгона после шиканы до подготовки к финальному сектору.
    Не захватываем 2860+, потому что там начинается anti-S подход.
    """
    d = S.get("distFromStart", 0.0)
    return 2550.0 < d < 3150.0


def is_finish_straight_zone(S):
    """
    V21: отключено.

    После шиканы и перед финальным поворотом больше нет отдельной прямой
    с anti-S логикой. Этот участок проходит как обычная часть трассы.
    """
    return False

# =====================================================================
# TARGET SPEED
# =====================================================================


def adaptive_target_speed(S, state):
    """
    V6 target speed.

    Берём разгон/потолок ближе к v3, но:
    - оставляем локальный cap на шикану;
    - сильнее страхуем последний поворот, где v5 вылетал;
    - front < 0 не убивает скорость в 0, чтобы recovery мог вернуть машину.
    """

    front = state["front"]
    front_delta = state["front_delta"]

    risk = state["total_risk"]
    speed_y = state["speed_y"]

    angle = abs(state["angle"])
    track_pos = abs(state["track_pos"])

    sector = get_sector_id(S)

    max_speed = ADAPT["max_speed"] * ADAPT["aggression"]
    sector_cap = SECTOR_SPEED_CAP.get(sector, ADAPT["max_speed"])
    max_speed = min(max_speed, sector_cap)

    # База почти как v3: быстрее на прямых и в поворотах.
    if front < 0:
        front_speed = 45.0
    elif front < 6:
        front_speed = 38.0
    elif front < 10:
        front_speed = 52.0
    elif front < 15:
        front_speed = 64.0
    elif front < 25:
        front_speed = 78.0
    elif front < 40:
        front_speed = 95.0
    elif front < 60:
        front_speed = 112.0
    elif front < 85:
        front_speed = 134.0
    elif front < 120:
        front_speed = 164.0
    elif front < 160:
        front_speed = 188.0
    else:
        front_speed = max_speed

    # Risk не должен срезать разгон на прямых.
    risk_speed = max_speed - risk * 7.0
    target_speed = min(max_speed, front_speed, risk_speed)

    # Закрытие поворота: мягко, без лишнего удушения.
    if front_delta < -10:
        target_speed = min(target_speed, 112.0)
    if front_delta < -22:
        target_speed = min(target_speed, 88.0)
    if front_delta < -38:
        target_speed = min(target_speed, 64.0)

    # Боковое скольжение.
    if speed_y > 8:
        target_speed = min(target_speed, 98.0)
    if speed_y > 13:
        target_speed = min(target_speed, 68.0)

    # Край трассы.
    if track_pos > 0.70:
        target_speed = min(target_speed, 105.0)
    if track_pos > 0.86:
        target_speed = min(target_speed, 70.0)
    if track_pos > 1.05:
        target_speed = min(target_speed, 48.0)

    # Угол.
    if angle > 0.32:
        target_speed = min(target_speed, 105.0)
    if angle > 0.52:
        target_speed = min(target_speed, 70.0)
    if angle > 0.75:
        target_speed = min(target_speed, 45.0)

    # Локальные страховки.
    target_speed = chicane_speed_cap(S, state, target_speed)
    target_speed = finish_corner_speed_cap(S, state, target_speed)

    # Минимальная скорость на трассе.
    on_track = track_pos < 0.98 and front > 0 and angle < 0.90
    if on_track:
        if front > 85:
            min_crawl_speed = 92.0
        elif front > 60:
            min_crawl_speed = 78.0
        elif front > 35:
            min_crawl_speed = 60.0
        elif front > 20:
            min_crawl_speed = 42.0
        elif front > 10:
            min_crawl_speed = 28.0
        else:
            min_crawl_speed = 0.0
        target_speed = max(target_speed, min_crawl_speed)

    # V26: после шиканы и до таблички 3 не продолжаем ехать медленно.
    # Если машина уже вернулась в коридор и впереди есть место — поднимаем target.
    d = S.get("distFromStart", 0.0)
    if 2590.0 < d < 3150.0 and track_pos < 0.62 and angle < 0.30 and front > 65:
        target_speed = max(target_speed, min(max_speed, 160.0))
    if 2590.0 < d < 3150.0 and track_pos < 0.42 and angle < 0.20 and front > 100:
        target_speed = max(target_speed, min(max_speed, 185.0))

    # V18: на шикане используем профиль скорости как нижнюю целевую скорость,
    # иначе front<6 превращал target в 38 и машина начинала резкий тормоз
    # прямо в момент поворота. Если машина уже почти за краем, страховка остаётся.
    d = S.get("distFromStart", 0.0)
    if 2300.0 <= d <= 2550.0 and state["front"] > 0 and abs(state["track_pos"]) < 0.95:
        profile_speed = chicane_profile_speed(d)
        if state["front"] < 5:
            profile_speed = min(profile_speed, 68.0)
        if abs(state["track_pos"]) > 0.72:
            profile_speed = min(profile_speed, 66.0)
        target_speed = max(target_speed, profile_speed)

    # V23 hard local caps are applied again at the very end.
    # This prevents min_crawl_speed and straight boosts from overriding
    # the chicane/final speed restrictions.
    target_speed = chicane_speed_cap(S, state, target_speed)
    target_speed = finish_corner_speed_cap(S, state, target_speed)

    return clamp(target_speed, 0.0, ADAPT["max_speed"])

# =====================================================================
# STEERING
# =====================================================================


def adaptive_steer(S, state):
    """
    V6 steering.

    Сохраняет chicane assist из v5, добавляет finish assist,
    а в обычных поворотах даёт больше стабильности без жёсткого
    "вжимания" машины в центр.
    """

    global PREV_STEER

    speed = state["speed"]
    angle = state["angle"]
    track_pos = state["track_pos"]
    front = state["front"]
    risk = state["total_risk"]
    dist = S.get("distFromStart", 0.0)

    # ---------------------------------------------------------
    # LOW SPEED STEERING
    # ---------------------------------------------------------
    if speed < 8:
        dist_raced = S.get("distRaced", 0.0)
        target_track_pos = 0.30 if dist_raced < 25 else 0.0
        pos_error = track_pos - target_track_pos
        target_steer = angle * 1.10 - pos_error * 0.45
        target_steer = clamp(target_steer, -0.14, 0.14)

        diff = target_steer - PREV_STEER
        diff = clamp(diff, -0.030, 0.030)
        PREV_STEER = clamp(PREV_STEER + diff, -0.14, 0.14)
        return PREV_STEER

    # ---------------------------------------------------------
    # CHICANE GENERAL-CORNER ASSIST — V19
    # ---------------------------------------------------------
    if is_chicane_zone(S):
        # Important: no S-profile and no feedforward steering here.
        # We handle the chicane like a normal slow corner: stay in a safe
        # corridor and use angle + trackPos correction with damping.
        target_pos = 0.0
        pos_error = track_pos - target_pos
        signed_speed_y = S.get("speedY", 0.0)

        target_steer = (
            angle * 1.18
            - pos_error * 0.92
            - signed_speed_y * 0.006
        )

        # Stronger return from edges. Sign follows the ordinary controller:
        # positive trackPos -> steer negative, negative trackPos -> steer positive.
        if track_pos > 0.38:
            target_steer -= 0.12
        if track_pos > 0.55:
            target_steer -= 0.22
        if track_pos > 0.72:
            target_steer -= 0.34

        if track_pos < -0.38:
            target_steer += 0.12
        if track_pos < -0.55:
            target_steer += 0.22
        if track_pos < -0.72:
            target_steer += 0.34

        # When the front sensors close, behave like other corners: more center correction.
        if front < 42:
            target_steer -= track_pos * 0.22
        if front < 26:
            target_steer -= track_pos * 0.34
        if front < 14:
            target_steer -= track_pos * 0.48

        if speed > 115:
            steer_limit = 0.34
        elif speed > 85:
            steer_limit = 0.44
        elif speed > 55:
            steer_limit = 0.56
        else:
            steer_limit = 0.64

        target_steer = clamp(target_steer, -steer_limit, steer_limit)

        if speed > 110:
            max_step = 0.050
        elif speed > 75:
            max_step = 0.070
        else:
            max_step = 0.095

        diff = target_steer - PREV_STEER
        diff = clamp(diff, -max_step, max_step)

        PREV_STEER = clamp(PREV_STEER + diff, -0.68, 0.68)
        return PREV_STEER

    # ---------------------------------------------------------
    # FINISH SECTOR ASSIST — V20 baseline-like anti-S
    # ---------------------------------------------------------
    if is_finish_corner_zone(S):
        target_pos = finish_line_target(dist)
        pos_error = track_pos - target_pos
        signed_speed_y = S.get("speedY", 0.0)

        # До 3245 м это НЕ поворот в центр, а стабильная правая/внешняя
        # полоса. Поэтому не тянем trackPos к 0.0.
        if dist < 3245.0:
            target_steer = (
                angle * 0.36
                - pos_error * 0.62
                - signed_speed_y * 0.005
            )

            # Anti-S: если боковая скорость уже большая, закрываем резкую
            # перекладку руля.
            if abs(signed_speed_y) > 5.5:
                target_steer *= 0.72
            if abs(signed_speed_y) > 8.0:
                target_steer *= 0.55

            if speed > 130:
                steer_limit = 0.18
            elif speed > 105:
                steer_limit = 0.22
            elif speed > 80:
                steer_limit = 0.28
            else:
                steer_limit = 0.34

            if speed > 120:
                max_step = 0.024
            elif speed > 90:
                max_step = 0.032
            else:
                max_step = 0.044

        else:
            # Сам финальный поворот: уже можно смещаться левее по профилю,
            # но без резкой S-перекладки.
            target_steer = (
                angle * 0.60
                - pos_error * 0.82
                - signed_speed_y * 0.006
            )

            if speed > 105:
                steer_limit = 0.34
            elif speed > 75:
                steer_limit = 0.44
            else:
                steer_limit = 0.54

            if speed > 100:
                max_step = 0.040
            elif speed > 70:
                max_step = 0.055
            else:
                max_step = 0.070

        target_steer = clamp(target_steer, -steer_limit, steer_limit)

        diff = target_steer - PREV_STEER
        diff = clamp(diff, -max_step, max_step)

        PREV_STEER = clamp(PREV_STEER + diff, -0.56, 0.56)
        return PREV_STEER

    # ---------------------------------------------------------
    # GENERAL STEERING
    # ---------------------------------------------------------
    target_steer = angle * ADAPT["steer_gain"] - track_pos * ADAPT["center_gain"] * 0.76

    # В закрытых местах возвращаемся, но не так резко, чтобы не было
    # "тормоз + выкрутить руль + газ".
    if front < 45:
        target_steer -= track_pos * 0.16
    if front < 25:
        target_steer -= track_pos * 0.26

    if track_pos > 0.70:
        target_steer -= 0.18
    if track_pos < -0.70:
        target_steer += 0.18
    if track_pos > 0.90:
        target_steer -= 0.34
    if track_pos < -0.90:
        target_steer += 0.34

    if speed > 145:
        steer_limit = 0.24
    elif speed > 120:
        steer_limit = 0.30
    elif speed > 85:
        steer_limit = 0.40
    elif front < 30:
        steer_limit = 0.54
    else:
        steer_limit = 0.50

    target_steer = clamp(target_steer, -steer_limit, steer_limit)

    if speed > 140:
        max_step = 0.038
    elif speed > 110:
        max_step = 0.050
    elif speed > 75:
        max_step = 0.068
    else:
        max_step = 0.088

    if risk > 0.78:
        max_step += 0.018

    diff = target_steer - PREV_STEER
    diff = clamp(diff, -max_step, max_step)

    PREV_STEER = clamp(PREV_STEER + diff, -0.70, 0.70)
    return PREV_STEER

# =====================================================================
# ACCEL / BRAKE
# =====================================================================


def adaptive_accel_brake(S, state, target_speed):
    """
    V6 accel/brake.

    Возвращаем резкий разгон v3, но убираем петлю recovery:
    - если выехали за трассу, сначала стабилизируем скорость и жёстко
      рулим обратно, а не делаем круг по траве;
    - на прямых держим сильный газ;
    - тормоз не должен мелко включаться там, где target выше speed.
    """

    global PREV_ACCEL, PREV_BRAKE

    speed = state["speed"]
    front = state["front"]
    risk = state["total_risk"]

    angle = abs(state["angle"])
    track_pos = state["track_pos"]
    abs_track_pos = abs(track_pos)
    speed_y = state["speed_y"]

    speed_error = target_speed - speed

    # ---------------------------------------------------------
    # OFFTRACK / EDGE RECOVERY
    # ---------------------------------------------------------
    # Если всё же вышли за предел, не делаем широкую петлю.
    # На высокой скорости коротко гасим, потом возвращаемся с малым газом.
    # ---------------------------------------------------------
    if front < 0 or abs_track_pos > 1.03:
        if speed > 62:
            PREV_ACCEL = 0.0
            PREV_BRAKE = 0.20
        elif speed > 42:
            PREV_ACCEL = 0.08
            PREV_BRAKE = 0.08
        elif speed > 22:
            PREV_ACCEL = 0.32
            PREV_BRAKE = 0.0
        else:
            PREV_ACCEL = 0.50
            PREV_BRAKE = 0.0
        return PREV_ACCEL, PREV_BRAKE

    # ---------------------------------------------------------
    # LOW SPEED CRAWL / RECOVERY
    # ---------------------------------------------------------
    if (
        speed < 8
        and front > 8
        and abs_track_pos < 1.05
        and angle < 0.95
    ):
        PREV_BRAKE = 0.0
        PREV_ACCEL = 0.52 if target_speed > 18 else 0.36
        return PREV_ACCEL, PREV_BRAKE

    accel = 0.0
    brake = 0.0

    # ---------------------------------------------------------
    # Emergency braking.
    # ---------------------------------------------------------
    if is_chicane_zone(S) and front < 5 and speed > 58:
        # V18: на самой шикане не бьём по тормозу 0.42.
        # Резкий тормоз + большой руль запускали вынос в борт.
        brake = 0.18
        accel = 0.0
    elif is_chicane_zone(S) and front < 9 and speed > 76:
        brake = 0.14
        accel = 0.0
    elif is_chicane_zone(S) and front < 16 and speed > 96:
        brake = 0.10
        accel = 0.0
    elif front < 5 and speed > 58:
        brake = 0.42
        accel = 0.0
    elif front < 9 and speed > 76:
        brake = 0.32
        accel = 0.0
    elif front < 16 and speed > 96:
        brake = 0.24
        accel = 0.0
    elif abs_track_pos > 0.92 and speed > 82:
        brake = 0.22
        accel = 0.0
    elif angle > 0.62 and speed > 72:
        brake = 0.24
        accel = 0.0
    elif speed_y > 14 and speed > 78:
        brake = 0.22
        accel = 0.0
    else:
        # Normal speed controller: v3-style stronger throttle.
        if speed_error < -45:
            brake = 0.26 * ADAPT["brake_gain"]
            accel = 0.0
        elif speed_error < -30:
            brake = 0.16 * ADAPT["brake_gain"]
            accel = 0.0
        elif speed_error < -15:
            brake = 0.07 * ADAPT["brake_gain"]
            accel = 0.0
        elif risk > 0.90 and speed > 55 and not is_chicane_zone(S) and not is_finish_corner_zone(S):
            brake = 0.04
            accel = 0.0
        else:
            brake = 0.0

            if speed_error > 45:
                accel = 1.00
            elif speed_error > 30:
                accel = 1.00
            elif speed_error > 18:
                accel = 0.92
            elif speed_error > 8:
                accel = 0.72
            elif speed_error > -3:
                accel = 0.34
            else:
                accel = 0.0

    brake = clamp(brake, 0.0, 1.0)

    if brake > 0.09:
        accel = 0.0

    # V9 STRAIGHT BOOST:
    # На прямых разгоняемся резче. Шикану и последний поворот не трогаем.
    dist = S.get("distFromStart", 0.0)

    if (
        not is_chicane_zone(S)
        and not is_finish_corner_zone(S)
        and front > 62
        and abs_track_pos < 0.62
        and angle < 0.26
        and target_speed > speed + 2
        and brake < 0.05
    ):
        if front > 110 and target_speed > speed + 6:
            accel = max(accel, 1.00)
        else:
            accel = max(accel, 0.96)

    # Отдельный буст стартовой прямой: раньше добираем скорость до первого торможения.
    if (
        18.0 <= dist <= 135.0
        and front > 75
        and abs_track_pos < 0.45
        and angle < 0.20
        and brake < 0.05
        and speed < 165
    ):
        accel = max(accel, 1.00)


    # V12 POST-CHICANE BOOST:
    # После шиканы сразу собираемся и разгоняемся, если машина в коридоре.
    if (
        is_post_chicane_straight(S)
        and not is_finish_corner_zone(S)
        and front > 60
        and abs_track_pos < 0.62
        and angle < 0.30
        and target_speed > speed + 2
    ):
        brake = 0.0
        if front > 95 and abs_track_pos < 0.46 and angle < 0.22:
            accel = max(accel, 1.00)
        else:
            accel = max(accel, 0.94)

    # У края не вжимаем газ.
    if is_chicane_zone(S) and abs_track_pos > 0.58 and speed > 38:
        accel = 0.0
    if is_finish_corner_zone(S) and abs_track_pos > 0.48 and speed > 38:
        accel = 0.0
    if (not is_chicane_zone(S)) and (not is_finish_corner_zone(S)) and abs_track_pos > 0.90 and speed > 75:
        accel = 0.0

    accel = clamp(accel, 0.0, 1.0)

    # Быстрый набор скорости, как v3.
    if accel > PREV_ACCEL:
        PREV_ACCEL = min(accel, PREV_ACCEL + 0.46)
    else:
        PREV_ACCEL = max(accel, PREV_ACCEL - 0.55)

    # Тормоз отпускаем быстро, чтобы не висеть на тормозе после поворота.
    if brake > PREV_BRAKE:
        PREV_BRAKE = min(brake, PREV_BRAKE + 0.10)
    else:
        PREV_BRAKE = max(brake, PREV_BRAKE - 0.50)

    return PREV_ACCEL, PREV_BRAKE

# =====================================================================
# GEARS
# =====================================================================


def adaptive_shift_gears(S):
    speed = S.get("speedX", 0.0)

    # V6: раньше включаем 6-ю, чтобы не упираться в 5 передачу на 140+.
    if speed < 28:
        return 1
    elif speed < 52:
        return 2
    elif speed < 82:
        return 3
    elif speed < 108:
        return 4
    elif speed < 140:
        return 5
    else:
        return 6

# =====================================================================
# LOGGING
# =====================================================================

def log_adaptive_run(S, R, state, target_speed):
    global LOG_INITIALIZED

    row = {
        "step": ADAPT_STEP,
        "distFromStart": S.get("distFromStart", 0.0),
        "distRaced": S.get("distRaced", 0.0),
        "curLapTime": S.get("curLapTime", 0.0),
        
        "rpm": S.get("rpm", 0.0),

        "wheelSpinVel_0": S.get("wheelSpinVel", [0, 0, 0, 0])[0],
        "wheelSpinVel_1": S.get("wheelSpinVel", [0, 0, 0, 0])[1],
        "wheelSpinVel_2": S.get("wheelSpinVel", [0, 0, 0, 0])[2],
        "wheelSpinVel_3": S.get("wheelSpinVel", [0, 0, 0, 0])[3],

        "sector": get_sector_id(S),
        "sector_cap": SECTOR_SPEED_CAP.get(get_sector_id(S), ADAPT["max_speed"]),

        "max_speed_param": ADAPT["max_speed"],
        "steer_gain": ADAPT["steer_gain"],
        "center_gain": ADAPT["center_gain"],
        "brake_gain": ADAPT["brake_gain"],

        "speedX": S.get("speedX", 0.0),
        "speedY": S.get("speedY", 0.0),
        "speedZ": S.get("speedZ", 0.0),

        "angle": S.get("angle", 0.0),
        "trackPos": S.get("trackPos", 0.0),

        "front": state["front"],
        "front_delta": state["front_delta"],

        "target_speed": target_speed,
        "steer": R.get("steer", 0.0),
        "accel": R.get("accel", 0.0),
        "brake": R.get("brake", 0.0),
        "gear": R.get("gear", 0),

        "risk": state["total_risk"],
        "aggression": ADAPT["aggression"],

        "target_trackPos": (
            chicane_line_target(S.get("distFromStart", 0.0))
            if is_chicane_zone(S) else
            0.0
        ),
        "mode": (
            "chicane" if is_chicane_zone(S) else
            "final_boards_321" if 3150.0 <= S.get("distFromStart", 0.0) <= 3294.0 else
            "post_final_boost" if 3294.0 < S.get("distFromStart", 0.0) < 3605.0 else
            "post_chicane" if is_post_chicane_straight(S) else
            "normal"
        ),

        "front_risk": state["front_risk"],
        "closing_risk": state["closing_risk"],
        "edge_risk": state["edge_risk"],
        "angle_risk": state["angle_risk"],
        "slide_risk": state["slide_risk"],

        
    }

    track = get_track(S)

    for i in range(19):
        row[f"track_{i}"] = track[i]


    mode = "a" if LOG_INITIALIZED else "w"

    with open(LOG_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not LOG_INITIALIZED:
            writer.writeheader()
            LOG_INITIALIZED = True

        writer.writerow(row)


# =====================================================================
# MAIN DRIVER
# =====================================================================

def drive_adaptive(c):
    """
    Главная функция управления машиной.
    """

    global ADAPT_STEP

    S, R = c.S.d, c.R.d

    state = analyze_state(S)
    update_adaptive_parameters(S, state)

    target_speed = adaptive_target_speed(S, state)
    steer = adaptive_steer(S, state)
    accel, brake = adaptive_accel_brake(S, state, target_speed)

    R["steer"] = steer
    R["accel"] = accel
    R["brake"] = brake
    R["gear"] = adaptive_shift_gears(S)

    # ---------------------------------------------------------
    # V19 CHICANE SAFETY LAYER
    # ---------------------------------------------------------
    dist = S.get("distFromStart", 0.0)

    # No second steering controller here. Steering is only adaptive_steer().
    # This layer only controls gas/brake so it cannot fight the steering logic.
    if 2180.0 <= dist <= 2550.0:
        tp = state["track_pos"]
        speed = state["speed"]
        signed_speed_y = abs(S.get("speedY", 0.0))

        # In the dangerous part, do not accelerate unless the car is calm.
        if 2300.0 <= dist <= 2500.0:
            if abs(tp) < 0.36 and signed_speed_y < 6.0 and speed < target_speed - 4:
                R["accel"] = min(max(R["accel"], 0.24), 0.42)
                R["brake"] = min(R["brake"], 0.04)
            else:
                R["accel"] = min(R["accel"], 0.10)
                if speed > target_speed + 8:
                    R["brake"] = max(R["brake"], 0.10)
        else:
            # Approach and exit: moderate gas only.
            R["accel"] = min(R["accel"], 0.42)
            if speed > target_speed + 10:
                R["brake"] = max(R["brake"], 0.10)

        # Edge protection.
        if abs(tp) > 0.50:
            R["accel"] = 0.0
            if speed > 54:
                R["brake"] = max(R["brake"], 0.16)
        if abs(tp) > 0.68:
            R["accel"] = 0.0
            if speed > 42:
                R["brake"] = max(R["brake"], 0.24)

    # V20: после шиканы сохраняем разгон только до 2860 м.
    # Дальше начинается финальный anti-S коридор: не тянем машину к центру,
    # а держим baseline-like линию через finish_line_target().
    if is_post_chicane_straight(S):
        tp = state["track_pos"]
        speed = state["speed"]
        signed_speed_y = S.get("speedY", 0.0)
        front = state["front"]

        straight_steer = clamp(
            state["angle"] * 0.24 - tp * 0.26 - signed_speed_y * 0.003,
            -0.14,
            0.14,
        )
        R["steer"] = clamp(R["steer"] * 0.82 + straight_steer * 0.18, -0.22, 0.22)

        if front > 80 and abs(tp) < 0.52 and abs(state["angle"]) < 0.24:
            R["brake"] = min(R["brake"], 0.04)
            if target_speed > speed + 5:
                R["accel"] = max(R["accel"], 0.86)

    if is_finish_straight_zone(S):
        tp = state["track_pos"]
        speed = state["speed"]
        signed_speed_y = S.get("speedY", 0.0)
        front = state["front"]
        target_pos = finish_line_target(dist)
        pos_error = tp - target_pos

        straight_steer = clamp(
            state["angle"] * 0.20 - pos_error * 0.38 - signed_speed_y * 0.003,
            -0.14,
            0.14,
        )
        R["steer"] = clamp(R["steer"] * 0.84 + straight_steer * 0.16, -0.22, 0.22)

        # На подходе к финалу не даём full gas, если началась раскачка.
        if 2860.0 <= dist < 3245.0:
            if abs(pos_error) > 0.28 or abs(signed_speed_y) > 5.5:
                R["accel"] = min(R["accel"], 0.24)
                if speed > target_speed + 8:
                    R["brake"] = max(R["brake"], 0.08)
            elif front > 70 and target_speed > speed + 5:
                R["brake"] = min(R["brake"], 0.04)
                R["accel"] = max(R["accel"], 0.72)

    # ---------------------------------------------------------
    # V26 FINAL SHARP CORNER: braking-board speed control
    # ---------------------------------------------------------
    # No S-corridor steering. We only force speed low enough to enter
    # the sharp final corner, then full acceleration after it.
    if 3150.0 <= dist <= 3294.0:
        speed = state["speed"]
        tp = state["track_pos"]
        angle = abs(state["angle"])
        signed_speed_y = abs(S.get("speedY", 0.0))

        final_cap = finish_corner_speed_cap(S, state, 999.0)

        # Hard braking when above local cap.
        if speed > final_cap + 22:
            R["accel"] = 0.0
            R["brake"] = max(R["brake"], 0.48)
        elif speed > final_cap + 14:
            R["accel"] = 0.0
            R["brake"] = max(R["brake"], 0.38)
        elif speed > final_cap + 7:
            R["accel"] = 0.0
            R["brake"] = max(R["brake"], 0.26)
        elif speed > final_cap + 2:
            R["accel"] = min(R["accel"], 0.08)
            R["brake"] = max(R["brake"], 0.12)
        else:
            # Before the apex keep throttle small; once the car is pointed out, release.
            if dist < 3288.0:
                R["accel"] = min(R["accel"], 0.18)
                R["brake"] = min(R["brake"], 0.04)
            else:
                if state["front"] > 38 and abs(tp) < 0.70 and angle < 0.60:
                    R["brake"] = 0.0
                    R["accel"] = 1.0

        # If unstable, keep it slow.
        if abs(tp) > 0.48 or angle > 0.40 or signed_speed_y > 7.0:
            R["accel"] = 0.0
            if speed > 45:
                R["brake"] = max(R["brake"], 0.30)

    # After the last sharp corner: maximum acceleration if the car is back on track.
    if 3294.0 < dist < 3605.0:
        if state["front"] > 20 and abs(state["track_pos"]) < 0.92 and abs(state["angle"]) < 0.88:
            R["brake"] = 0.0
            R["accel"] = 1.00

    if False and 2860.0 <= dist <= 3360.0:
        tp = state["track_pos"]
        speed = state["speed"]
        signed_speed_y = S.get("speedY", 0.0)
        target_pos = finish_line_target(dist)
        pos_error = tp - target_pos

        # Не делаем вторую сильную коррекцию руля к центру.
        # Только мягко подмешиваем тот же target_pos, чтобы не было спора блоков.
        corridor_correction = clamp(
            state["angle"] * 0.16 - pos_error * 0.32 - signed_speed_y * 0.003,
            -0.16,
            0.16,
        )
        R["steer"] = clamp(R["steer"] * 0.82 + corridor_correction * 0.18, -0.30, 0.30)

        if 2860.0 <= dist < 3245.0:
            if abs(pos_error) > 0.34:
                R["accel"] = min(R["accel"], 0.20)
                if speed > 88:
                    R["brake"] = max(R["brake"], 0.10)
            if abs(pos_error) > 0.50:
                R["accel"] = 0.0
                if speed > 70:
                    R["brake"] = max(R["brake"], 0.16)
            if abs(signed_speed_y) > 8.0:
                R["accel"] = min(R["accel"], 0.12)
                if speed > 80:
                    R["brake"] = max(R["brake"], 0.12)
        else:
            if abs(pos_error) > 0.38:
                R["accel"] = min(R["accel"], 0.18)
                if speed > 78:
                    R["brake"] = max(R["brake"], 0.14)
            if abs(pos_error) > 0.58:
                R["accel"] = 0.0
                if speed > 58:
                    R["brake"] = max(R["brake"], 0.22)

    # ---------------------------------------------------------
    # FINAL RECOVERY LAYER
    # ---------------------------------------------------------
    # Последний защитный слой после всех расчётов.
    # Если машина колесом вышла за трассу, она не должна стоять
    # с brake=0.6. Она должна мягко вернуться обратно.
    # ---------------------------------------------------------
    if state["front"] < 0 or abs(state["track_pos"]) > 1.03:
        tp = state["track_pos"]
        speed = state["speed"]
        dist = S.get("distFromStart", 0.0)

        # V7 LOCAL FIX ONLY:
        # Вылет был в шикане около 2452 м влево. Там старое recovery
        # давало только steer=0.42, этого не хватало: машина уходила
        # до trackPos -2.0 и разворачивалась. Усиливаем recovery только
        # для этой зоны.
        if 2180.0 <= dist <= 2550.0:
            # V19: recovery uses the same sign as the ordinary center return.
            # positive trackPos -> steer negative; negative trackPos -> steer positive.
            if tp > 0:
                R["steer"] = -0.58
            else:
                R["steer"] = 0.58

            if speed > 48:
                R["accel"] = 0.0
                R["brake"] = 0.30
            elif speed > 25:
                R["accel"] = 0.12
                R["brake"] = 0.04
            else:
                R["accel"] = 0.36
                R["brake"] = 0.0

            R["gear"] = 2 if speed > 35 else 1

        elif 3150.0 <= dist <= 3294.0:
            # V26 final-corner recovery: slow down and steer back to the track.
            if tp > 0:
                R["steer"] = -0.58
            else:
                R["steer"] = 0.58

            if speed > 50:
                R["accel"] = 0.0
                R["brake"] = 0.36
            elif speed > 22:
                R["accel"] = 0.10
                R["brake"] = 0.08
            else:
                R["accel"] = 0.34
                R["brake"] = 0.0

            R["gear"] = 2 if speed > 35 else 1

        else:
            # Руль в сторону центра.
            if tp > 0:
                R["steer"] = -0.42
            else:
                R["steer"] = 0.42

            if speed > 55:
                R["accel"] = 0.0
                R["brake"] = 0.22
            elif speed > 18:
                R["accel"] = 0.24
                R["brake"] = 0.0
            else:
                R["accel"] = 0.45
                R["brake"] = 0.0

            R["gear"] = 2 if speed > 35 else 1

    ADAPT_STEP += 1

    log_adaptive_run(S, R, state, target_speed)

    if ADAPT_STEP % 25 == 0:
        print(
            "step =", ADAPT_STEP,
            "ADAPTIVE_V27",
            "speed =", round(state["speed"], 1),
            "target =", round(target_speed, 1),
            "front =", round(state["front"], 1),
            "front_delta =", round(state["front_delta"], 1),
            "trackPos =", round(state["track_pos"], 2),
            "angle =", round(state["angle"], 3),
            "risk =", round(state["total_risk"], 2),
            "aggr =", round(ADAPT["aggression"], 2),
            "steer =", round(R["steer"], 3),
            "accel =", round(R["accel"], 2),
            "brake =", round(R["brake"], 2),
            "gear =", R["gear"],
        )

# =====================================================================
# MAIN LOOP
# =====================================================================

if __name__ == "__main__":
    C = Client(p=3001)

    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_adaptive(C)
        C.respond_to_server()

    C.shutdown()