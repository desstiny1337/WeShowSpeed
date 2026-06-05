
import socket
import sys
import getopt
import os
import time
import csv

print("!!! THIS FILE IS RUNNING MEOW1 !!!")

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



#############################################
# MODULAR DRIVE LOGIC WITH USER PARAMETERS  #
#############################################

import math

# ================= USER CONFIGURABLE PARAMETERS =================
TARGET_SPEED = 190  # Target speed in km/h. Increasing this makes the car go faster but may reduce stability.
STEER_GAIN = 22    # Steering sensitivity. Higher values make the car turn more aggressively.
CENTERING_GAIN = 0.28  # How strongly the car corrects its position toward the center of the track.
BRAKE_THRESHOLD = 0.30  # Angle threshold for braking. Lower values brake earlier.
GEAR_SPEEDS = [0, 40, 80, 120, 165, 215]  # Speed thresholds for gear shifting.
ENABLE_TRACTION_CONTROL = False  # Toggle traction control system.

# ================= HELPER FUNCTIONS =================
PREV_STEER = 0.0

def calculate_steering(S):
    global PREV_STEER

    raw_steer = (S['angle'] * STEER_GAIN / math.pi) - (S['trackPos'] * CENTERING_GAIN)
    raw_steer = max(-0.45, min(0.45, raw_steer))

    max_change = 0.07

    if raw_steer > PREV_STEER + max_change:
        steer = PREV_STEER + max_change
    elif raw_steer < PREV_STEER - max_change:
        steer = PREV_STEER - max_change
    else:
        steer = raw_steer

    PREV_STEER = steer
    return steer

def calculate_throttle(S, R):
    front = S['track'][9]
    angle = abs(S['angle'])
    speedZ = S.get('speedZ', 0)
    trackPos = S.get('trackPos', 0)

    # Если почти вне трассы — газ не даём
    if abs(trackPos) > 1.05:
        return 0.0

    # Если машина почти у края на высокой скорости — газ убираем
    if abs(trackPos) > 0.95 and S['speedX'] > 70:
        return 0.0

    # Если машину реально сильно боком — газ не даём
    if angle > 0.85 and S['speedX'] > 45:
        return 0.0

    # Если совсем мало места впереди — не газуем
    if front < 22 and S['speedX'] > 55:
        return 0.0

    # На спуске газ режем только в реально опасной близкой зоне
    if speedZ < -3.0 and front < 35 and abs(R['steer']) > 0.35 and S['speedX'] > 50:
        return 0.0

    target_speed = calculate_target_speed(S)

    # Exit boost: открываем газ раньше после шиканы / медленной зоны
    if front > 55 and S['speedX'] < 95 and angle < 0.45 and abs(trackPos) < 0.70:
        return 1.0

    # Ещё один общий выход из поворота:
    # если скорость ниже цели, машина ровная и есть место — газуем
    if S['speedX'] < target_speed - 3 and front > 35 and angle < 0.55 and abs(trackPos) < 0.85:
        return 1.0

    # Обычный режим газа
    target_speed -= abs(R['steer']) * 4

    if S['speedX'] < target_speed:
        accel = 1.0
    else:
        accel = 0.40

    if S['speedX'] < 20:
        accel = 0.9

    return max(0.0, min(1.0, accel))

PREV_BRAKE = 0.0

def smooth_brake(target_brake):
    global PREV_BRAKE

    press_speed = 0.10
    release_speed = 0.75

    if target_brake > PREV_BRAKE:
        brake = min(target_brake, PREV_BRAKE + press_speed)
    else:
        brake = max(target_brake, PREV_BRAKE - release_speed)

    PREV_BRAKE = brake
    return brake

def apply_brakes(S):
    speed = S['speedX']
    angle = abs(S['angle'])
    track = S['track']
    speedZ = S.get('speedZ', 0)
    trackPos = S.get('trackPos', 0)

    front = track[9]
    left_front = max(track[3:9])
    right_front = max(track[10:16])

    side_diff = abs(left_front - right_front)
    corner_sharpness = max(left_front, right_front) - front

    target_speed = calculate_target_speed(S)
    overspeed = speed - target_speed

    # 1. Авария / потеряли трассу
    if front < 0 and speed > 35:
        return 0.70

    # 2. Если почти вне трассы — мягко тормозим
    if abs(trackPos) > 1.05 and speed > 45:
        return 0.35

    # 3. Если почти у края — совсем мягкий тормоз
    if abs(trackPos) > 0.95 and speed > 75:
        return 0.20

    # 4. Самая опасная часть шиканы на спуске
    if speedZ < -3.0 and front < 30 and speed > 65:
        return 0.65

    # 5. На выходе из шиканы не тормозим,
    # если машина ровная и скорость ещё низкая
    if front > 60 and speed < 95 and angle < 0.35 and abs(trackPos) < 0.65:
        return 0.0

    # 6. На выходе из поворота не дёргаем тормоз,
    # если машина едет медленнее или около целевой скорости
    if front > 40 and speed < target_speed + 5 and angle < 0.5:
        return 0.0

    # 7. Если скорость ниже целевой — не тормозим
    if speed < target_speed - 5:
        return 0.0

    # 8. Спуск + близкий поворот
    if speedZ < -3.0 and front < 55 and speed > 90:
        return 0.45

    # 9. Настоящая резкая шикана
    if front < 50 and side_diff > 35 and corner_sharpness > 18 and speed > 75:
        return 0.60

    # 10. Очень мало места впереди
    if front < 30 and speed > 70:
        return 0.60

    # 11. Обычное превышение скорости
    if overspeed > 45:
        return 0.45
    elif overspeed > 30:
        return 0.25
    elif overspeed > 18:
        return 0.10

    # 12. Если машину реально несёт боком
    if angle > 1.0:
        return 0.20

    return 0.0

def calculate_target_speed(S):
    track = S['track']
    speedZ = S.get('speedZ', 0)

    front = track[9]
    left_front = max(track[3:9])
    right_front = max(track[10:16])

    side_diff = abs(left_front - right_front)
    corner_sharpness = max(left_front, right_front) - front

    # Потеряли трассу
    if front < 0:
        return 55

    # Самая опасная часть шиканы на спуске
    if speedZ < -3.0 and front < 30:
        return 65

    # Спуск + близкий поворот
    if speedZ < -3.0 and front < 55:
        return 85

    # Выход из шиканы / медленной зоны:
    # если машина уже ровная и впереди есть место — разрешаем быстрее разгоняться
    if front > 60 and abs(S['angle']) < 0.35 and abs(S['trackPos']) < 0.65 and S['speedX'] < 95:
        return 100

    # Спуск, но ещё можно ехать
    if speedZ < -3.0 and front < 90:
        return 105

    # Настоящая резкая шикана
    if front < 50 and side_diff > 35 and corner_sharpness > 18:
        return 70

    # Обычные повороты
    if front < 55:
        return 85

    if front < 85:
        return 115

    if front < 120:
        return 145

    if front < 160:
        return 165

    return TARGET_SPEED

PREV_GEAR = 1

def shift_gears(S):
    speed = S['speedX']

    if speed < 30:
        return 1
    elif speed < 50:
        return 2
    elif speed < 85:
        return 3
    elif speed < 130:
        return 4
    elif speed < 180:
        return 5
    else:
        return 6

def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL:
        if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])) > 2:
            accel -= 0.1
    return max(0.0, accel)

#================TELEMETRY================


LOG_FILE = "telemetry_log.csv"
LOG_INITIALIZED = False
STEP_COUNTER = 0

def log_telemetry(S, R, target_speed, target_brake):
    global LOG_INITIALIZED, STEP_COUNTER

    track = S['track']

    row = {
        "step": STEP_COUNTER,
        "speedX": S.get("speedX", 0),
        "speedY": S.get("speedY", 0),
        "speedZ": S.get("speedZ", 0),
        "angle": S.get("angle", 0),
        "trackPos": S.get("trackPos", 0),
        "z": S.get("z", 0),
        "rpm": S.get("rpm", 0),
        "gear": R.get("gear", 0),
        "steer": R.get("steer", 0),
        "accel": R.get("accel", 0),
        "brake": R.get("brake", 0),
        "target_speed": target_speed,
        "target_brake": target_brake,
        "front": track[9],
        "left_front": max(track[3:9]),
        "right_front": max(track[10:16]),
        "track_0": track[0],
        "track_1": track[1],
        "track_2": track[2],
        "track_3": track[3],
        "track_4": track[4],
        "track_5": track[5],
        "track_6": track[6],
        "track_7": track[7],
        "track_8": track[8],
        "track_9": track[9],
        "track_10": track[10],
        "track_11": track[11],
        "track_12": track[12],
        "track_13": track[13],
        "track_14": track[14],
        "track_15": track[15],
        "track_16": track[16],
        "track_17": track[17],
        "track_18": track[18],
    }

    mode = "a" if LOG_INITIALIZED else "w"

    with open(LOG_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not LOG_INITIALIZED:
            writer.writeheader()
            LOG_INITIALIZED = True

        writer.writerow(row)

    STEP_COUNTER += 1





# ================= MAIN DRIVE FUNCTION =================
def drive_modular(c):
    S, R = c.S.d, c.R.d

    front = S['track'][9]
    angle = abs(S['angle'])

    # Если машина уже развернулась/едет назад — не газовать в пол
    if S['speedX'] < -2 or angle > 2.2 or front < -0.5:
        R['brake'] = 0.0
        R['accel'] = 0.25
        R['gear'] = 1

        if S['angle'] > 0:
            R['steer'] = -0.35
        else:
            R['steer'] = 0.35

        return

    R['steer'] = calculate_steering(S)

    target_brake = apply_brakes(S)
    R['brake'] = smooth_brake(target_brake)

    trackPos = S.get('trackPos', 0)

    # Safety: машина реально уходит к краю / в гравий
    if abs(trackPos) > 0.95 and S['speedX'] > 55:
        R['accel'] = 0.0
        R['brake'] = max(R['brake'], 0.15)

        # Рулим обратно к центру трассы
        if trackPos > 0:
            R['steer'] = min(R['steer'], -0.18)
        else:
            R['steer'] = max(R['steer'], 0.18)

        R['gear'] = shift_gears(S)
        return

    # Anti-slide: если машину начало разворачивать, стабилизируем
    if abs(S['angle']) > 0.75 and S['speedX'] > 25:
        R['accel'] = 0.0
        R['brake'] = 0.15

        # Контрруление против заноса
        if S['angle'] > 0:
            R['steer'] = -0.25
        else:
            R['steer'] = 0.25

        R['gear'] = shift_gears(S)
        return

    # Если почти стоим, но ещё на трассе — выезжаем
    if S['speedX'] < 15 and front > 10 and angle < 0.8:
        R['brake'] = 0.0
        R['accel'] = 0.8
        R['gear'] = 1
    else:
                # Если тормоз сильный — газ не даём
        if R['brake'] > 0.20:
            R['accel'] = 0.0
        else:
            R['accel'] = calculate_throttle(S, R)
            R['accel'] = traction_control(S, R['accel'])

        R['gear'] = shift_gears(S)

    current_target_speed = calculate_target_speed(S)

    wheel_slip = (
        (S['wheelSpinVel'][2] + S['wheelSpinVel'][3])
        - (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])
    )

    "target_speed =", round(current_target_speed, 1),
    "slip =", round(wheel_slip, 1),
    "rpm =", round(S.get('rpm', 0), 0),

    current_target_speed = calculate_target_speed(S)
    log_telemetry(S, R, current_target_speed, target_brake)

    print(
        "target_speed =", round(current_target_speed, 1),
        "target_brake =", round(target_brake, 2),
        "real_brake =", round(R['brake'], 2),
        "accel =", round(R['accel'], 2),
        "gear =", R['gear'],
        "speed =", round(S['speedX'], 1),
        "front =", round(front, 1),
        "steer =", round(R['steer'], 2),
        "angle =", round(S['angle'], 2),
        "speedZ =", round(S['speedZ'], 2),
        "trackPos =", round(S['trackPos'], 2),
    )

    return

# ================= MAIN LOOP =================
if __name__ == "__main__":
    C = Client(p=3001)
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_modular(C)
        C.respond_to_server()
    C.shutdown()
