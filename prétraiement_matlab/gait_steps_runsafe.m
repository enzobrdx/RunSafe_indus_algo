function [outputStruct] = gait_steps_runsafe(neutral,dynamic,angles,velocities,hz,plots)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Loads the NEUTRAL and DYNAMIC data structures, divides the ANGLES and 
%   VELOCITIES data structures into the different left and right steps and
%   provides an index of touchdown and toeoffs (EVENTS/EVENT). The function 
%   also outputs time normalized angles (NORM_ANG) and velocities 
%   (NORM_VEL), gait speed (SPEEDOUTPUT), the method of  touchdown/toe off 
%   detection (EVENTSFLAG), variables of interest (DISCRETE_VARIABLES)
%   and the determined gait type (LABEL).
%
%  INPUTS
%  --------
%  NEUTRAL (struct):    Marker shell positions collected as part of 
%                       the static trial.
%
%  DYNAMIC (stuct):     Marker shell positions collected as part of
%                       the dynamic (run/walk) trial.
%
%  ANGLES (struct):     Angles (joint angles) structure created as an 
%                       output from the function: gait_kinematics.
%
%  VELOCITIES (struct): Velocities (joint velocities) structure created as 
%                       an output from the function: gait_kinematics.
%
%  HZ (int):        Data collection sampling frequency.
%
%  PLOTS (bool):    Boolean selected to generate plotted outcomes if 
%                   desired. If no second argument exists, or if PLOTS == 0
%                   , the plotted outputs in this function are suppressed.
%
%
%  OUTPUTS
%  -------
%  NORM_ANG (struct):   Normalized angles from touchown to takeoff across 
%                       all retained steps.
%
%  NORM_VEL (struct):   Normalized velocities from touchown to takeoff 
%                       across  all retained steps.
%
%  EVENTS (mat):    Matrix of frame numbers for touchdown and toeoffs.
%
%  EVENT (mat):     Same matric as EVENTS but also includes midswing in
%                   order to calculate swing variables
%
%  DISCRETE_VARIABLES (mat):    Contains the peaks and values of interest 
%                               for reporting.
%
%  SPEEDOUTPUT (float): Calculated speed of lowest heel marker. 
%
%  EVENTSFLAG (mat):Matrix the same size as EVENTS which indicates whether
%                   PCA event detection was used (1) or if the default FF 
%                   and FB events were used (0).
%
%  LABEL (str):     Returns a string which indicates whether the trial was 
%                   a 'walk' or 'run' based on the classifier in this 
%                   function.
%
%  LICENSE
%  -------
%  See file LICENSE.txt
%
% Copyright (C) 2010-2023,  Blayne Hettinga, Sean Osis, Allan Brett and
%                           The Running Injury Clinic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%for troubleshooting

% neutral = out.neutral;
% % dynamic = out.walking;
% dynamic = out.running;
% angles  = r_angles;
% velocities = r_velocities;
% hz = 200;
% plots = 0;

%% if no indication to plot the data was specified, default to not plot
if (nargin < 7)
    plots = 0;
end

%% Determine functional measures and gait type (walk vs run)
% movement speed comes from the A/P position time history of a heel marker
% so we first need to identify a heel marker

% Combine 3 of the foot markers into one matrix (ignore the created fourth)
L_foot = [neutral.L_foot_1;neutral.L_foot_2;neutral.L_foot_3];
% sort the markers from left to right
[L_foot,i_lf] = sortrows(L_foot,1);
% find the lower of the two medial markers
if L_foot(2,2) < L_foot(3,2)
    L_marker = strcat('L_foot_',num2str(i_lf(2)));
    L_heel = dynamic.(L_marker);
else
    L_marker = strcat('L_foot_',num2str(i_lf(3)));
    L_heel = dynamic.(L_marker);
end

feature accel off
[pks,~] = findpeaks(diff(L_heel(:,3)),'minpeakdistance',round(0.5*hz),'minpeakheight',0);
feature accel on

vel = hz*median(pks)/1000;


feature accel off
[~,locs] = findpeaks(-diff(L_heel(:,3)),'minpeakdistance',round(0.5*hz),'minpeakheight',0);
feature accel on

stRate = 60/(median(diff(locs))/hz);

speedoutput=vel;


% RIGHT SIDE
% Combine 3 of the foot markers into one matrix (ignore the created fourth)
R_foot = [neutral.R_foot_1;neutral.R_foot_2;neutral.R_foot_3];
% sort the markers from left to right
[R_foot,i_rf] = sortrows(R_foot,1);
% find the lower of the two medial markers
if R_foot(1,2) < R_foot(2,2)
    R_marker = strcat('R_foot_',num2str(i_rf(1)));
    R_heel = dynamic.(R_marker);
else
    R_marker = strcat('R_foot_',num2str(i_rf(2)));
    R_heel = dynamic.(R_marker);
end


% Identify gait type using a trained LDA classifier.  This will be more
% robust for shuffle-runners, older adults and speed walkers.  gaitClass
% represents an LDA object which has been trained on 839 test sets of
% walking and running, and validated on ~2000 sets of walking and running.
testSet = [vel stRate];
import classreg.learning.classif.CompactClassificationDiscriminant
load('gaitClass.mat','gaitClass')


label = predict(gaitClass,testSet);
%label returned as cell
label = label{1};


%% Identify Touch Down and Take Off events: Gait Independent

if plots~=0
    if strcmp(label,'walk')
        disp(['The subject was WALKING at ' num2str(vel) 'm/s or ' num2str(vel*3.6/1.6) 'mph' ]); disp(' ');
    else
        disp(['The subject was RUNNING at ' num2str(vel) 'm/s or ' num2str(vel*3.6/1.6) 'mph' ]); disp(' ');
        
    end
end


% Use PCA touchdown detection based on updated Osis et al. (2014) for
% both walking and running.
% Use new PCA toeoff detection for both walking and running.
% evt variables are NOT rounded

feature accel off
try
    [evtLtd,evtRtd] = pca_td(angles,hz,label);
    [evtLto,evtRto] = pca_to(angles,hz,label);
catch ME
    %For a small number of people, these functions return errors, or in the
    %case of bad data... default to use FF and FB in these cases
    
    evtLtd = [];
    evtRtd = [];
    evtLto = [];
    evtRto = [];
    
    disp('Automated event detection failed, defaulting to foot-forward foot-back')
    
    ME.message;
end
feature accel on


%% LEFT FOOT EVENTS

% when the feet are not tracked very well, discontinuities in the heel
% marker can occur causing the findpeaks to pick up additional 'peaks'
% for the purposes of simply identifying foot forward and foot back
% timing, we can over filter this signal. We do not care about the
% magnitude of the signal but only the timing so we can overfit as long
% as the filter has a zero phase shift.
% Note: signal is now filtered by default.  There is no advantage to not
% filtering, as if the signal quality is already good, then the system uses
% PCA event detection anyhow, and if the signal is bad, then it has to be
% filtered in order to get foot-forward foot-backward events.

feature accel off

[B,A] = butter(2, 5/(hz/2), 'low');
filtered_L_heel = filtfilt(B,A,L_heel(:,3));

% Begin by creating a gross estimation of foot forwards and foot backs
[~,L_FFi] = findpeaks(-filtered_L_heel(:,1),'minpeakdistance',round(0.35*hz));

if strcmp(label,'walk')
    % Use peak foot flexion angle for foot back
    % To deal with peaks resulting from signal flipping, threshold them
    angSig = angles.L_foot(:,3);
    angSig(abs(angSig)>90,:) = nan;
    [~,L_FBi] = findpeaks(-angSig,'minpeakdistance',round(0.7*hz),'minpeakheight',20);
else
    % Use rearmost position of heel marker for foot back
    [~,L_FBi] = findpeaks(filtered_L_heel(:,1),'minpeakdistance',round(0.35*hz));
end
feature accel on

%Uncomment block below to enable more aggressive quality control of data

% if prctile(abs(angles.L_foot(:,3)),90) > 120 && vel < 4
%
% error('Left ankle values outside of expected ranges, please ensure your shoe markers are properly placed and redo your collection')
%
% end


% Remove any leading FB
L_FBi(L_FBi<L_FFi(1)) = [];

%% find largest chunk of continuous data

%We want to check before and after that there is sufficient data for
%analysis
if size(L_FFi,1) < 2 || size(L_FBi,1) < 2
    error('Automated event detection unable to pull adequate number of strides for analysis. Please redo your data collection.')
end


[L_FFi, L_FBi, L_block_start, ~]=largest_block(L_FFi, L_FBi);

if size(L_FFi,1) < 2 || size(L_FBi,1) < 2
    error('Automated event detection unable to pull adequate number of strides for analysis. Please redo your data collection.')
end

%% TOUCHDOWN
% evtLtd from above


% SELECT SEQUENTIAL STEPS

% create an ordered set of sequential steps using FFi as guide
closest = abs(repmat(L_FFi(:),1,length(evtLtd))-repmat(evtLtd(:)',length(L_FFi),1));
[mindist,minx] = nanmin(closest,[],1);
for i = unique(minx)
    if sum(ismember(minx,i))>1
        mindist(minx==i) = [];
        evtLtd(minx==i) = [];
        minx(minx==i) = [];
    end
end

% Parameter based on the typical frame adjustments observed in 300
% datasets
if strcmp(label,'run')
    maxadj = 0.05*hz;
else
    maxadj = 0.10*hz;
end


% Preallocate
L_TD = nan(length(L_FFi),1);
evFltd = zeros(length(L_FFi),1);

% Here we replace FF indices with indices from evt where criteria are
% met... the default is to use FF
for i = 1:length(L_FFi)
    try
        
        if i > max(minx)
            break
        elseif ismember(i,minx) && mindist(minx==i)<maxadj
            % Replace with evtLtd since its more accurate
            L_TD(i) = evtLtd(minx==i);
            evFltd(i) = 1;
        else
            % Use FFi since it is more robust
            L_TD(i) = L_FFi(i);
        end
        
    catch err
        err.message;
        L_TD(i) = L_FFi(i);
    end
end


% TAKEOFF

% evtLto from above

% SELECT SEQUENTIAL STEPS

% Now create an ordered set of sequential steps using FBi as guide
closest = abs(repmat(L_FBi(:),1,length(evtLto))-repmat(evtLto(:)',length(L_FBi),1));

[mindist,minx] = nanmin(closest,[],1);
for i = unique(minx)
    if sum(ismember(minx,i))>1
        mindist(minx==i) = [];
        evtLto(minx==i) = [];
        minx(minx==i) = [];
    end
end

%Parameter based on the frame adjustment observed from 300 datasets
maxadj = 0.15*hz;

% Preallocate
L_TO = nan(length(L_FBi),1);
evFlto = zeros(length(L_FBi),1);

% Here we replace FB indices with TO from PCA default is to use FB
for i = 1:length(L_FBi)
    try
        
        if i > max(minx)
            break
        elseif ismember(i,minx) && mindist(minx==i)<maxadj
            % Replace with evtLto since its more accurate
            L_TO(i) = evtLto(minx==i);
            evFlto(i) = 1;
        else
            % Use FBi since it is more robust
            L_TO(i) = L_FBi(i);
        end
        
    catch err
        err.message;
        L_TO(i) = L_FBi(i);
    end
end

% Finally we round to get final indices
L_TD = round(L_TD);
L_TO = round(L_TO);



%% RIGHT FOOT EVENTS
% the same steps we just took for the left side

% Begin by creating a gross estimation of foot forwards and foot backs
feature accel off

[B,A] = butter(2, 5/(hz/2), 'low');
filtered_R_heel = filtfilt(B,A,R_heel(:,3));

[~,R_FFi] = findpeaks(-filtered_R_heel(:,1),'minpeakdistance',round(0.35*hz));

if strcmp(label,'walk')
    % To deal with peaks that result from signal flipping, threshold them
    angSig = angles.R_foot(:,3);
    angSig(abs(angSig)>90,:) = nan;
    [~,R_FBi] = findpeaks(-angSig,'minpeakdistance',round(0.7*hz),'minpeakheight',20);
else
    [~,R_FBi] = findpeaks(filtered_R_heel(:,1),'minpeakdistance',round(0.35*hz));
end

feature accel on

%Uncomment block below to enable more aggressive quality control of data

% if prctile(abs(angles.R_foot(:,3)),90) > 120 && vel < 4
%
% error('Right ankle values outside of expected ranges, please ensure your shoe markers are properly placed and redo your collection')
%
% end

% Remove any leading FF and FB
R_FFi(R_FFi<L_FFi(1)) = [];
R_FBi(R_FBi<R_FFi(1)) = [];

%% find largest block of continuous data

%We want to check before and after that there is sufficient data for
%analysis
if size(R_FFi,1) < 2 || size(R_FBi,1) < 2
    error('Automated event detection unable to pull adequate number of strides for analysis. Please redo your data collection.')
end

[R_FFi, R_FBi, R_block_start, ~]=largest_block(R_FFi, R_FBi);

if size(R_FFi,1) < 2 || size(R_FBi,1) < 2
    error('Automated event detection unable to pull adequate number of strides for analysis. Please redo your data collection.')
end

%In rare instances a the index will be in incorrect order run below again
%in case

% Remove any leading FF and FB
R_FFi(R_FFi<L_FFi(1)) = [];
R_FBi(R_FBi<R_FFi(1)) = [];

%%

% TOUCHDOWN
% evtRtd from above


% SELECT SEQUENTIAL STEPS

% Now create an ordered set of sequential steps using above elements
closest = abs(repmat(R_FFi(:),1,length(evtRtd))-repmat(evtRtd(:)',length(R_FFi),1));
[mindist,minx] = nanmin(closest,[],1);
for i = unique(minx)
    if sum(ismember(minx,i))>1
        mindist(minx==i) = [];
        evtRtd(minx==i) = [];
        minx(minx==i) = [];
    end
end

% Parameter based on the typical frame adjustment observed for 300
% datasets
if strcmp(label,'run')
    maxadj = 0.05*hz;
else
    maxadj = 0.10*hz;
end


R_TD = nan(length(R_FFi),1);
evFrtd = zeros(length(R_FFi),1);

% Here we replace FF indices with evt indices where criteria are met
% default is to use FF
for i = 1:length(R_FFi)
    try
        
        if i > max(minx)
            break
        elseif ismember(i,minx) && mindist(minx==i)<maxadj
            % Replace with evtRtd since its more accurate
            R_TD(i) = evtRtd(minx==i);
            evFrtd(i) = 1;
        else
            % Use FFi since it is more robust
            R_TD(i) = R_FFi(i);
        end
        
    catch err
        err.message;
        R_TD(i) = R_FFi(i);
    end
end


% TAKEOFF

%evtRto from above

% SELECT SEQUENTIAL STEPS

% Now create an ordered set of sequential steps using above elements
closest = abs(repmat(R_FBi(:),1,length(evtRto))-repmat(evtRto(:)',length(R_FBi),1));
[mindist,minx] = nanmin(closest,[],1);
for i = unique(minx)
    if sum(ismember(minx,i))>1
        mindist(minx==i) = [];
        evtRto(minx==i) = [];
        minx(minx==i) = [];
    end
end


%Parameter based on the frame adjustment observed from 300 datasets
maxadj = 0.15*hz;

R_TO = nan(length(R_FBi),1);
evFrto = zeros(length(R_FBi),1);

% Here we replace FB indices with TO from PCA default is to use FB
for i = 1:length(R_FBi)
    try
        
        if i > max(minx)
            break
        elseif ismember(i,minx) && mindist(minx==i)<=maxadj
            % Replace with evtRto since its more accurate
            R_TO(i) = evtRto(minx==i);
            evFrto(i) = 1;
        else
            % Use FBi since it is more robust
            R_TO(i) = R_FBi(i);
        end
        
    catch err
        err.message;
        R_TO(i) = R_FBi(i);
    end
end

% Finally, round to get final indices
R_TD = round(R_TD);
R_TO = round(R_TO);


%% if largest chunk of continuous data not at beginning, chop both right and left so they match


%index must begin with left touchdown and end with right toe
%off

if R_block_start < L_block_start
    %remove all right indices that occur before the first left touchdown
    R_TO((R_TD(:,1)<L_block_start)==1,:)=[];
    R_TD((R_TD(:,1)<L_block_start)==1,:)=[];
end

%end

flag = 0;

if L_block_start < R_block_start
    %remove left touchdowns more than one touchdown before the first right touchdown
    cut_inds = (L_TD(:,1)<R_block_start)==1;
    %this loop ensures the first index will be a left touchdown
    for i = 1:size(cut_inds,1)
        if cut_inds(i) == 1 && cut_inds(i+1) == 0 && flag ==0
            cut_inds(i) = 0;
            flag = 1;
        end
    end
    
    L_TD(cut_inds,:) = [];
    L_TO(cut_inds,:) = [];
    clear cut_inds
end


%% create an events matrix

% Remove trailing nans that may have crept in
evFltd(isnan(L_TD)) = [];
evFlto(isnan(L_TO)) = [];
evFrtd(isnan(R_TD)) = [];
evFrto(isnan(R_TO)) = [];

L_TD(isnan(L_TD)) = [];
L_TO(isnan(L_TO)) = [];
R_TD(isnan(R_TD)) = [];
R_TO(isnan(R_TO)) = [];

% Find the closest ordered pairs of L_TO and R_TD to synchronize steps
closest = abs(repmat(R_TD(:),1,length(L_TO))-repmat(L_TO(:)',length(R_TD),1));
[~,minx] = nanmin(closest,[],1);

% Truncate right stances to match up with left
evFrtd = evFrtd(unique(minx));
R_TD = R_TD(unique(minx));

testlength = min([length(L_TO) length(R_TD)]);
if median(L_TO(1:testlength)-R_TD(1:testlength))<0 % Then there is a flight phase
    
    % Find the closest ordered pairs of R_TD and R_TO to synchronize steps
    closest = abs(repmat(R_TO(:),1,length(R_TD))-repmat(R_TD(:)',length(R_TO),1));
    [~,minx] = nanmin(closest,[],1);
    evFrto = evFrto(unique(minx));
    R_TO = R_TO(unique(minx));
    
else % There is no flight phase i.e. grounded running or walking
    
    % Find the closest ordered pairs of R_TO and L_TD to synchronize steps
    tmp = L_TD(2:end);
    closest = abs(repmat(R_TO(:),1,length(tmp))-repmat(tmp(:)',length(R_TO),1));
    [~,minx] = nanmin(closest,[],1);
    evFrto = evFrto(unique(minx));
    R_TO = R_TO(unique(minx));
    
end


events = [length(L_TD),length(L_TO),length(R_TD),length(R_TO)];

% Chop everything to the same length
L_TD = L_TD(1:min(events));
L_TO = L_TO(1:min(events));
R_TD = R_TD(1:min(events));
R_TO = R_TO(1:min(events));

evFltd = evFltd(1:min(events));
evFlto = evFlto(1:min(events));
evFrtd = evFrtd(1:min(events));
evFrto = evFrto(1:min(events));

% Very rarely, these will wind up empty and assignment doesn't work
if isempty(L_TD) || isempty(L_TO) || isempty(R_TD) || isempty(R_TO)
    % skip
else
    events = nan(min(events),4);
    events(:,1)=L_TD;
    events(:,2)=L_TO;
    events(:,3)=R_TD;
    events(:,4)=R_TO;
end


% Very rarely, these will wind up empty and assignment doesn't work
if isempty(evFltd) || isempty(evFlto) || isempty(evFrtd) || isempty(evFrto)
    % skip
else
    eventsflag(:,1) = evFltd;
    eventsflag(:,2) = evFlto;
    eventsflag(:,3) = evFrtd;
    eventsflag(:,4) = evFrto;
end


% Remove first row since these will very often be reliant on FF and FB
% measures
if size(events,1) > 1
    events(1,:) = [];
    eventsflag(1,:) = [];
end


% Occasionally, one stance will drop out, and data becomes
% discontinuous...this fix alleviates this by trimming data to largest
% continuous block
try
    cont = [events(2:end,1)>events(1:end-1,2) events(2:end,3)>events(1:end-1,4)];  %Touchdowns have to come before Toeoffs for same leg
    F = find(any([0 0;cont;0 0]==0,2));
    D = diff(F)-2;
    [M,L] = max(D);
    events = events(F(L):F(L)+M,:);
    eventsflag = eventsflag(F(L):F(L)+M,:);
catch ME
    disp('Could not obtain a continuous block of events')
    events = [];
    eventsflag = [];
    ME.message;
end


% Worst-case... return to foot forward, foot back detection
if size(events,1) < 5
    disp('Automated event detection failed, defaulting to foot-forward foot-back')
    
    events = [length(L_FFi),length(L_FBi),length(R_FFi),length(R_FBi)];
    
    % Chop everything to the same length
    L_FFi = L_FFi(1:min(events));
    L_FBi = L_FBi(1:min(events));
    R_FFi = R_FFi(1:min(events));
    R_FBi = R_FBi(1:min(events));
    
    events = nan(min(events),4);
    events(:,1)=L_FFi;
    events(:,2)=L_FBi;
    events(:,3)=R_FFi;
    events(:,4)=R_FBi;
    
    eventsflag = zeros(size(events));
    
end


% Pull event columns from events so everything is consistent
L_TD = events(:,1);
L_TO = events(:,2);
R_TD = events(:,3);
R_TO = events(:,4);


%% Normalize the steps 0 to 100 from touchdown to takeoff
% do this by using a cubic spline filling function

% preallocate for speed
norm_ang.L_ankle = zeros(101,length(L_TD),3);
norm_ang.L_knee = norm_ang.L_ankle;
norm_ang.L_hip = norm_ang.L_ankle;
norm_ang.L_foot = norm_ang.L_ankle;
norm_ang.L_pelvis = norm_ang.L_ankle;
norm_vel.L_ankle = norm_ang.L_ankle;
norm_vel.L_knee = norm_ang.L_ankle;
norm_vel.L_hip = norm_ang.L_ankle;
norm_vel.L_pelvis = norm_ang.L_ankle;
norm_pos.L_heel = norm_ang.L_ankle;

norm_ang.R_ankle = zeros(101,length(R_TD),3);
norm_ang.R_knee = norm_ang.R_ankle;
norm_ang.R_hip = norm_ang.R_ankle;
norm_ang.R_foot = norm_ang.R_ankle;
norm_ang.R_pelvis = norm_ang.R_ankle;
norm_vel.R_ankle = norm_ang.R_ankle;
norm_vel.R_knee = norm_ang.R_ankle;
norm_vel.R_hip = norm_ang.R_ankle;
norm_vel.R_pelvis = norm_ang.R_ankle;
norm_pos.R_heel = norm_ang.R_ankle;


for i=1:length(L_TD)
    norm_ang.L_ankle(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),angles.L_ankle(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    norm_ang.L_knee(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),angles.L_knee(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    norm_ang.L_hip(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),angles.L_hip(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    
    norm_ang.L_foot(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),angles.L_foot(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    norm_ang.L_pelvis(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),angles.pelvis(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    
    norm_vel.L_ankle(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),velocities.L_ankle(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    norm_vel.L_knee(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),velocities.L_knee(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    norm_vel.L_hip(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),velocities.L_hip(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    norm_vel.L_pelvis(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),velocities.pelvis(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    
    norm_pos.L_heel(:,i,:) = interp1(0:(L_TO(i)-L_TD(i)),L_heel(L_TD(i):L_TO(i),:),0:(L_TO(i)-L_TD(i))/100:(L_TO(i)-L_TD(i)),'pchip');
    
end
for i=1:length(R_TD)
    norm_ang.R_ankle(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),angles.R_ankle(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    norm_ang.R_knee(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),angles.R_knee(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    norm_ang.R_hip(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),angles.R_hip(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    
    norm_ang.R_foot(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),angles.R_foot(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    norm_ang.R_pelvis(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),angles.pelvis(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    
    norm_vel.R_ankle(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),velocities.R_ankle(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    norm_vel.R_knee(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),velocities.R_knee(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    norm_vel.R_hip(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),velocities.R_hip(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    norm_vel.R_pelvis(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),velocities.pelvis(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
    
    norm_pos.R_heel(:,i,:) = interp1(0:(R_TO(i)-R_TD(i)),R_heel(R_TD(i):R_TO(i),:),0:(R_TO(i)-R_TD(i))/100:(R_TO(i)-R_TD(i)),'pchip');
end

%% in order to identify the heelwhip, we want the foot angle in the global
% projected angle of the long axis of the foot into the floor
% during the swing phase from takeoff to touchdown
L_foot_angle = zeros(101,length(L_TD)-1,3);
for i=1:length(L_TD)-1
    L_foot_angle(:,i,:) = interp1(0:(L_TD(i+1)-L_TO(i)), angles.L_foot(L_TO(i):L_TD(i+1),:), 0:(L_TD(i+1)-L_TO(i))/100:(L_TD(i+1)-L_TO(i)), 'pchip');
end
R_foot_angle = zeros(101,length(R_TD)-1,3);
for i=1:length(R_TD)-1
    R_foot_angle(:,i,:) = interp1(0:(R_TD(i+1)-R_TO(i)), angles.R_foot(R_TO(i):R_TD(i+1),:), 0:(R_TD(i+1)-R_TO(i))/100:(R_TD(i+1)-R_TO(i)), 'pchip');
end

%% use the 'DROP THE BAD' function to seperate the good data

% Specify whether curves are plotted
drop_plot = 1;

[norm_ang.L_ankle,norm_ang.R_ankle]   = drop_the_bad(norm_ang.L_ankle,norm_ang.R_ankle,drop_plot);
[norm_ang.L_knee,norm_ang.R_knee]     = drop_the_bad(norm_ang.L_knee,norm_ang.R_knee,drop_plot);
[norm_ang.L_hip,norm_ang.R_hip]       = drop_the_bad(norm_ang.L_hip,norm_ang.R_hip,drop_plot);
[norm_ang.L_foot,norm_ang.R_foot]     = drop_the_bad(norm_ang.L_foot,norm_ang.R_foot,drop_plot);
[norm_ang.L_pelvis,norm_ang.R_pelvis] = drop_the_bad(norm_ang.L_pelvis,norm_ang.R_pelvis,drop_plot);
[norm_vel.L_ankle,norm_vel.R_ankle]   = drop_the_bad(norm_vel.L_ankle,norm_vel.R_ankle,drop_plot);
[norm_vel.L_knee,norm_vel.R_knee]     = drop_the_bad(norm_vel.L_knee,norm_vel.R_knee,drop_plot);
[norm_vel.L_hip,norm_vel.R_hip]       = drop_the_bad(norm_vel.L_hip,norm_vel.R_hip,drop_plot);
[norm_vel.L_pelvis,norm_vel.R_pelvis] = drop_the_bad(norm_vel.L_pelvis,norm_vel.R_pelvis,drop_plot);
[norm_pos.L_heel,norm_pos.R_heel]     = drop_the_bad(norm_pos.L_heel,norm_pos.R_heel,drop_plot);

close(findobj('tag', 'drop_the_bad_temp_figure'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MODIFICATION RUNSAFE: Calcul de la médiane et de l'écart-type
%
% La matrice 'DISCRETE_VARIABLES' est remplacée par 'outputStruct'
% pour ne calculer que les 10 variables requises, en utilisant les 
% formules exactes du code original.

outputStruct = struct();

% --- 1. 'STRIDE_RATE' (Cadence) ---
% Formule originale: sort(60*hz./diff(L_TD(:)))
temp_L = 60*hz./diff(L_TD(:));
outputStruct.L_STRIDE_RATE_median = median(temp_L);
outputStruct.L_STRIDE_RATE_std = std(temp_L);

temp_R = 60*hz./diff(R_TD(:));
outputStruct.R_STRIDE_RATE_median = median(temp_R);
outputStruct.R_STRIDE_RATE_std = std(temp_R);

% --- 2. 'STRIDE_LENGTH' (Longueur de foulée) ---
% Formule originale: sort((vel)*diff(L_TD(:))./hz)
temp_L = (vel)*diff(L_TD(:))./hz;
outputStruct.L_STRIDE_LENGTH_median = median(temp_L);
outputStruct.L_STRIDE_LENGTH_std = std(temp_L);

temp_R = (vel)*diff(R_TD(:))./hz;
outputStruct.R_STRIDE_LENGTH_median = median(temp_R);
outputStruct.R_STRIDE_LENGTH_std = std(temp_R);

% --- 3. 'SWING_TIME' (Temps de vol) ---
% Formule originale: (L_TD(i+1)-L_TO(i))/hz
L_swing_frames = nan(1,length(L_TD)-1);
for i = 1:length(L_TD)-1
    L_swing_frames(i)=L_TD(i+1)-L_TO(i);
end
temp_L = (L_swing_frames)/hz;
outputStruct.L_SWING_TIME_median = nanmedian(temp_L); % Utilise nanmedian comme l'original
outputStruct.L_SWING_TIME_std = nanstd(temp_L);

R_swing_frames = nan(1,length(R_TD)-1);
for i = 1:length(R_TD)-1
    R_swing_frames(i)=R_TD(i+1)-R_TO(i);
end
temp_R = (R_swing_frames)/hz;
outputStruct.R_SWING_TIME_median = nanmedian(temp_R); % Utilise nanmedian comme l'original
outputStruct.R_SWING_TIME_std = nanstd(temp_R);

% --- 4. 'STANCE_TIME' (Temps de contact au sol) ---
% Formule originale: (L_TO-L_TD)/hz
temp_L = (L_TO-L_TD)/hz;
outputStruct.L_STANCE_TIME_median = nanmedian(temp_L); % Utilise nanmedian comme l'original
outputStruct.L_STANCE_TIME_std = nanstd(temp_L);

temp_R = (R_TO-R_TD)/hz;
outputStruct.R_STANCE_TIME_median = nanmedian(temp_R); % Utilise nanmedian comme l'original
outputStruct.R_STANCE_TIME_std = nanstd(temp_R);

% --- 5. 'ANKLE_DF_PEAK_ANGLE' (Pic de Dorsiflexion) ---
% Formule originale: - median(min(norm_ang.L_ankle(:,:,3)))
temp_L = squeeze(min(norm_ang.L_ankle(:,:,3),[],1));
outputStruct.L_ANKLE_DF_PEAK_ANGLE_median = -median(temp_L);
outputStruct.L_ANKLE_DF_PEAK_ANGLE_std = std(temp_L);

temp_R = squeeze(min(norm_ang.R_ankle(:,:,3),[],1));
outputStruct.R_ANKLE_DF_PEAK_ANGLE_median = -median(temp_R);
outputStruct.R_ANKLE_DF_PEAK_ANGLE_std = std(temp_R);

% --- 6. 'ANKLE_EVE_PEAK_ANGLE' (Pic d'Éversion) ---
% Formule originale L: median(min(norm_ang.L_ankle(:,:,1)))
temp_L = squeeze(min(norm_ang.L_ankle(:,:,1),[],1));
outputStruct.L_ANKLE_EVE_PEAK_ANGLE_median = median(temp_L);
outputStruct.L_ANKLE_EVE_PEAK_ANGLE_std = std(temp_L);

% Formule originale R: - median(max(norm_ang.R_ankle(:,:,1)))
temp_R = squeeze(max(norm_ang.R_ankle(:,:,1),[],1));
outputStruct.R_ANKLE_EVE_PEAK_ANGLE_median = -median(temp_R);
outputStruct.R_ANKLE_EVE_PEAK_ANGLE_std = std(temp_R);

% --- 7. 'ANKLE_ROT_PEAK_ANGLE' (Pic de Rotation Cheville) ---
% Formule originale L: median(min(norm_ang.L_ankle(:,:,2)))
temp_L = squeeze(min(norm_ang.L_ankle(:,:,2),[],1));
outputStruct.L_ANKLE_ROT_PEAK_ANGLE_median = median(temp_L);
outputStruct.L_ANKLE_ROT_PEAK_ANGLE_std = std(temp_L);

% Formule originale R: - median(max(norm_ang.R_ankle(:,:,2)))
temp_R = squeeze(max(norm_ang.R_ankle(:,:,2),[],1));
outputStruct.R_ANKLE_ROT_PEAK_ANGLE_median = -median(temp_R);
outputStruct.R_ANKLE_ROT_PEAK_ANGLE_std = std(temp_R);

% --- 8. 'FOOT_PROG_ANGLE' (Angle de Progression du Pied) ---
% Formule originale L: - median(mean(norm_ang.L_foot(20:50,:,2)))
temp_L = squeeze(mean(norm_ang.L_foot(20:50,:,2),1));
outputStruct.L_FOOT_PROG_ANGLE_median = -median(temp_L);
outputStruct.L_FOOT_PROG_ANGLE_std = std(temp_L);

% Formule originale R: - median(mean(norm_ang.R_foot(20:50,:,2)))
temp_R = squeeze(mean(norm_ang.R_foot(20:50,:,2),1));
outputStruct.R_FOOT_PROG_ANGLE_median = -median(temp_R);
outputStruct.R_FOOT_PROG_ANGLE_std = std(temp_R);

% --- 9. 'FOOT_ANG_at_HS' (Angle du Pied au Contact) ---
% Formule originale L: median(norm_ang.L_foot(1,:,3))
temp_L = squeeze(norm_ang.L_foot(1,:,3));
outputStruct.L_FOOT_ANG_at_HS_median = median(temp_L);
outputStruct.L_FOOT_ANG_at_HS_std = std(temp_L);

% Formule originale R: median(norm_ang.R_foot(1,:,3))
temp_R = squeeze(norm_ang.R_foot(1,:,3));
outputStruct.R_FOOT_ANG_at_HS_median = median(temp_R);
outputStruct.R_FOOT_ANG_at_HS_std = std(temp_R);

% --- 10. 'VERTICAL_OSCILLATION' (Oscillation Verticale) ---
filtered_pelvis = filtfilt(B,A,dynamic.pelvis_4);
try
    % 'plots' est mis à 0 pour le traitement par lots
    [vertical_oscillation] = oscillation(filtered_pelvis, L_TD, L_TO, R_TD, R_TO, 0, label);
    
    % Formule originale L: median(vertical_oscillation(:,2))
    temp_L = vertical_oscillation(:,2);
    outputStruct.L_VERTICAL_OSCILLATION_median = nanmedian(temp_L);
    outputStruct.L_VERTICAL_OSCILLATION_std = nanstd(temp_L);
    
    % Formule originale R: median(vertical_oscillation(:,4))
    temp_R = vertical_oscillation(:,4);
    outputStruct.R_VERTICAL_OSCILLATION_median = nanmedian(temp_R);
    outputStruct.R_VERTICAL_OSCILLATION_std = nanstd(temp_R);

catch ME
    warning('Erreur lors du calcul de l''oscillation verticale: %s');
    % Assure que les champs existent même en cas d'erreur
    outputStruct.L_VERTICAL_OSCILLATION_median = NaN;
    outputStruct.L_VERTICAL_OSCILLATION_std = NaN;
    outputStruct.R_VERTICAL_OSCILLATION_median = NaN;
    outputStruct.R_VERTICAL_OSCILLATION_std = NaN;
end

% --- 11. 'PRONATION_ONSET' (% Stance) ---
% Calcul basé sur le code original gait_steps

% Initialisation des vecteurs pour stocker les indices de chaque foulée
num_steps_L = size(norm_ang.L_ankle, 2);
L_pros = nan(1, num_steps_L); % Onset Gauche
L_sups = nan(1, num_steps_L); % Offset Gauche

for i = 1:num_steps_L % Boucle sur chaque foulée gauche nettoyée
    try
        step_data = norm_ang.L_ankle(:, i, 1); % Données d'eversion/inversion pour cette foulée
        if all(isnan(step_data)) % Ignorer si la foulée est pleine de NaN
             continue;
        end
        min_eve = min(step_data);
        max_eve = max(step_data);
        rom = max_eve - min_eve; % Amplitude du mouvement
        
        % Trouver les indices où l'angle est inférieur au pic + 20% ROM
        % (Début de pronation et fin de resupination)
        threshold = min_eve + 0.2 * rom;
        a = find(step_data < threshold);
        
        if ~isempty(a)
             L_pros(i) = a(1);  % Premier indice trouvé = Onset
             L_sups(i) = a(end); % Dernier indice trouvé = Offset
        end
    catch ME
        warning('Erreur calcul pronation onset/offset Gauche, foulée %d: %s', i, ME.message);
        % Laisse NaN si erreur
    end
end

% Calcul Médiane et Écart-Type pour Gauche
outputStruct.L_PRONATION_ONSET_median = round(nanmedian(L_pros)); % Arrondi comme l'original
outputStruct.L_PRONATION_ONSET_std = nanstd(L_pros);
outputStruct.L_PRONATION_OFFSET_median = round(nanmedian(L_sups)); % Arrondi comme l'original
outputStruct.L_PRONATION_OFFSET_std = nanstd(L_sups);

% --- Calcul pour le côté Droit ---
num_steps_R = size(norm_ang.R_ankle, 2);
R_pros = nan(1, num_steps_R); % Onset Droit
R_sups = nan(1, num_steps_R); % Offset Droit

for i = 1:num_steps_R % Boucle sur chaque foulée droite nettoyée
    try
        step_data = -norm_ang.R_ankle(:, i, 1); % Inversion signe pour Droite
         if all(isnan(step_data)) % Ignorer si la foulée est pleine de NaN
             continue;
        end
        min_eve = min(step_data); % Minimum après inversion = Pic d'éversion
        max_eve = max(step_data);
        rom = max_eve - min_eve;
        
        % Trouver les indices où l'angle (inversé) est inférieur au pic + 20% ROM
        threshold = min_eve + 0.2 * rom;
        a = find(step_data < threshold);

        if ~isempty(a)
            R_pros(i) = a(1);  % Premier indice trouvé = Onset
            R_sups(i) = a(end); % Dernier indice trouvé = Offset
        end
    catch ME
        warning('Erreur calcul pronation onset/offset Droit, foulée %d: %s', i, ME.message);
        % Laisse NaN si erreur
    end
end

% Calcul Médiane et Écart-Type pour Droit
outputStruct.R_PRONATION_ONSET_median = round(nanmedian(R_pros)); % Arrondi comme l'original
outputStruct.R_PRONATION_ONSET_std = nanstd(R_pros);
outputStruct.R_PRONATION_OFFSET_median = round(nanmedian(R_sups)); % Arrondi comme l'original
outputStruct.R_PRONATION_OFFSET_std = nanstd(R_sups);

% --- Fin de la section de calcul ---
close(findobj('tag', 'steps_temp_figure'));
end

function [L_out,R_out] = drop_the_bad(L_data,R_data,plots)

% ... (début de la fonction inchangé) ...

%% LEFT
% need an initial value that will be removed at the end
L_bad=0; L_good=0;
count = [0,length(L_bad)];
change = 1;
ml = mean(L_data,2);
sdl = std(L_data,1,2);

while change(end) > 0
    
    for j = 1:length(L_data(1,:,1)) % number of steps
        if sum(j == L_bad) == 1 % has the step already been flagged as 'bad'?
            break % if so, skip it
        else % otherwise, carry on
            for k = 1:3 % number of dimensions xyz
                for i = 1:length(L_data(:,1,k)) % normalized time
                    % check if the trial has already been marked as bad and skip it
                    if L_bad(end) == j
                        break
                    else
                        % is the value further than 3SD from the mean (above OR below)
                        if L_data(i,j,k) > ml(i,1,k) + 3*sdl(i,1,k) || L_data(i,j,k) < ml(i,1,k) - 3*sdl(i,1,k)
                            if j > L_bad(end) % if bad, collect the step if not already collected
                                L_bad = [L_bad,j];
                            end
                            break % no need to continue the loop if the step is bad
                        end
                    end
                end
                if k==3
                    if j > L_bad(end) % if the step is not bad, collect the step as good
                        L_good = [L_good,j];
                    end
                end
            end
        end
    end
    
    % remove the leading zero
    L_bad = L_bad(2:end);
    L_good = L_good(2:end);
    
    % an additional catch for the situation where there are two
    % groups of data that are very seperate such that 3*SD doesn't work
    % The data are split into groups using a histogram. If there are only 2
    % groups and they are the first and last, only keep the group with the most
    % data in it and get ride of the smaller group
    
    %% MODIFICATION: Ajout d'une vérification pour éviter l'erreur sur hist([])
    if ~isempty(L_good) 
        for i=1:length(L_data(:,1,1)) % for every normalized time point(1:101)
            for j=1:length(L_data(1,1,:)) % for every dimension xyz
                [n,bin]=hist(L_data(i,L_good,j)); % find the info to make a histogram
                if n(1)>0 && n(10)>0 && sum(n(2:9))< 1 % if there are data in the first, last, but one or less inbetween
                    if n(1)>n(10) % and if there are more data in the first bin than the last,
                        [~,b] = find(L_data(i,L_good,j) < bin(2)); % find the trials in the first bin
                        L_good = L_good(b); % and only keep the ones we want
                        [~,d] = find(L_data(i,L_good,j) > bin(2)); % id the trials in the last bin
                        L_bad = [L_bad,d]; % and get rid of them
                    else % or if there are more data in the last bin
                        [~,b] = find(L_data(i,L_good,j) > bin(9)); % find the trials in the last bin
                        L_good = L_good(b); % and only keep those ones
                        [~,d] = find(L_data(i,L_good,j) < bin(2)); % id the smaller group
                        L_bad = [L_bad,d]; % and ignore them
                    end
                    break
                end
            end
        end
    end
    
    % add the leading zeros back in ... had to be a better way to do this ...
    L_bad = [0,L_bad];
    L_good = [0,L_good];
    
    count = [count,length(L_bad)];
    change = diff(count);
    
    %% MODIFICATION: Ajout d'une vérification pour éviter l'erreur sur mean([])
    if length(L_good) > 1
        ml = mean(L_data(:,L_good(2:end),:),2);
        sdl = std(L_data(:,L_good(2:end),:),1,2);
    else
        % Si L_good est vide ou n'a que [0], on arrête la boucle
        change(end) = 0; 
    end
    
end

% and remove the leading zero
L_bad = L_bad(2:end);
L_good = L_good(2:end);


%%
%% RIGHT
% need an initial value that will be removed at the end
R_bad=0; R_good=0;
count = [0,length(R_bad)];
change = 1;
mr = mean(R_data,2);
sdr = std(R_data,1,2);

while change(end) > 0
    
    for j = 1:length(R_data(1,:,1)) % number of steps
        if sum(j == R_bad) == 1 % has the step already been flagged as 'bad'?
            break % if so, skip it
        else % otherwise, carry on
            for k = 1:3 % number of dimensions xyz
                for i = 1:length(R_data(:,1,k)) % normalized time
                    % check if the trial has already been marked as bad and skip it
                    if R_bad(end) == j
                        break
                    else
                        % is the value further than 3SD from the mean (above OR below)
                        if R_data(i,j,k) > mr(i,1,k) + 3*sdr(i,1,k) || R_data(i,j,k) < mr(i,1,k) - 3*sdr(i,1,k)
                            if j > R_bad(end) % if bad, collect the step if not already collected
                                R_bad = [R_bad,j];
                            end
                            break % no need to continue the loop if the step is bad
                        end
                    end
                end
                if k==3
                    if j > R_bad(end) % if the step is not bad, collect the step as good
                        R_good = [R_good,j];
                    end
                end
            end
        end
    end
    
    % remove the leading zero
    R_bad = R_bad(2:end);
    R_good = R_good(2:end);
    
    % an additional catch for the situation where there are two
    % groups of data that are very seperate such that 3*SD doesn't work
    % The data are split into groups using a histogram. If there are only 2
    % groups and they are the first and last, only keep the group with the most
    % data in it and get ride of the smaller group
    
    %% MODIFICATION: Ajout d'une vérification pour éviter l'erreur sur hist([])
    if ~isempty(R_good)
        for i=1:length(R_data(:,1,1)) % for every normalized time point(1:101)
            for j=1:length(R_data(1,1,:)) % for every dimension xyz
                [n,bin]=hist(R_data(i,R_good,j)); % find the info to make a histogram
                if n(1)>0 && n(10)>0 && sum(n(2:9))< 1 % if there are data in the first, last, but one or less inbetween
                    if n(1)>n(10) % and if there are more data in the first bin than the last,
                        [~,b] = find(R_data(i,R_good,j) < bin(2)); % find the trials in the first bin
                        R_good = R_good(b); % and only keep the ones we want
                        [~,d] = find(R_data(i,R_good,j) > bin(2)); % id the trials in the last bin
                        R_bad = [R_bad,d]; % and get rid of them
                    else % or if there are more data in the last bin
                        [~,b] = find(R_data(i,R_good,j) > bin(9)); % find the trials in the last bin
                        R_good = R_good(b); % and only keep those ones
                        [~,d] = find(R_data(i,R_good,j) < bin(2)); % id the smaller group
                        R_bad = [R_bad,d]; % and ignore them
                    end
                    break
                end
            end
        end
    end
    
    % add the leading zeros back in ... had to be a better way to do this ...
    R_bad = [0,R_bad];
    R_good = [0,R_good];
    
    count = [count,length(R_bad)];
    change = diff(count);
    
    %% MODIFICATION: Ajout d'une vérification pour éviter l'erreur sur mean([])
    if length(R_good) > 1
        mr = mean(R_data(:,R_good(2:end),:),2);
        sdr = std(R_data(:,R_good(2:end),:),1,2);
    else
        % Si R_good est vide ou n'a que [0], on arrête la boucle
        change(end) = 0;
    end
    
end

% and remove the leading zero
R_bad = R_bad(2:end);
R_good = R_good(2:end);


%% PLOTS
if isempty(L_good); LG = nan(101,1,3); else LG = L_data(:,L_good,:); end
if isempty(L_bad); LB = nan(101,1,3);  else LB = L_data(:,L_bad,:);  end
if isempty(R_good); RG = nan(101,1,3); else RG = R_data(:,R_good,:); end
if isempty(R_bad); RB = nan(101,1,3);  else RB = R_data(:,R_bad,:);  end

if plots ~= 0
    figure('tag','drop_the_bad_temp_figure'); hold on;
    subplot(321); plot(0:100,LG(:,:,1),'b');hold on;plot(0:100,LB(:,:,1),'r');title('SELECTED LEFT ANGLES - X')
    fill([0:100,flip(0:100,2)],[(ml(:,:,1)+sdl(:,:,1))',(flip((ml(:,:,1)-sdl(:,:,1)),1))'],[4 4 4]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(ml(:,:,1)+2*sdl(:,:,1))',(flip((ml(:,:,1)-2*sdl(:,:,1)),1))'],[5 5 5]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(ml(:,:,1)+3*sdl(:,:,1))',(flip((ml(:,:,1)-3*sdl(:,:,1)),1))'],[6 6 6]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    subplot(323); plot(0:100,LG(:,:,2),'b');hold on;plot(0:100,LB(:,:,2),'r');title('Y')
    fill([0:100,flip(0:100,2)],[(ml(:,:,2)+sdl(:,:,2))',(flip((ml(:,:,2)-sdl(:,:,2)),1))'],[4 4 4]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(ml(:,:,2)+2*sdl(:,:,2))',(flip((ml(:,:,2)-2*sdl(:,:,2)),1))'],[5 5 5]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(ml(:,:,2)+3*sdl(:,:,2))',(flip((ml(:,:,2)-3*sdl(:,:,2)),1))'],[6 6 6]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    subplot(325); plot(0:100,LG(:,:,3),'b');hold on;plot(0:100,LB(:,:,3),'r');title('Z')
    fill([0:100,flip(0:100,2)],[(ml(:,:,3)+sdl(:,:,3))',(flip((ml(:,:,3)-sdl(:,:,3)),1))'],[4 4 4]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(ml(:,:,3)+2*sdl(:,:,3))',(flip((ml(:,:,3)-2*sdl(:,:,3)),1))'],[5 5 5]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(ml(:,:,3)+3*sdl(:,:,3))',(flip((ml(:,:,3)-3*sdl(:,:,3)),1))'],[6 6 6]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    
    subplot(322); plot(0:100,RG(:,:,1),'b');hold on;plot(0:100,RB(:,:,1),'r');title('SELECTED RIGHT ANGLES - X')
    fill([0:100,flip(0:100,2)],[(mr(:,:,1)+sdr(:,:,1))',(flip((mr(:,:,1)-sdr(:,:,1)),1))'],[4 4 4]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(mr(:,:,1)+2*sdr(:,:,1))',(flip((mr(:,:,1)-2*sdr(:,:,1)),1))'],[5 5 5]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(mr(:,:,1)+3*sdr(:,:,1))',(flip((mr(:,:,1)-3*sdr(:,:,1)),1))'],[6 6 6]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    subplot(324); plot(0:100,RG(:,:,2),'b');hold on;plot(0:100,RB(:,:,2),'r');title('Y')
    fill([0:100,flip(0:100,2)],[(mr(:,:,2)+sdr(:,:,2))',(flip((mr(:,:,2)-sdr(:,:,2)),1))'],[4 4 4]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(mr(:,:,2)+2*sdr(:,:,2))',(flip((mr(:,:,2)-2*sdr(:,:,2)),1))'],[5 5 5]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(mr(:,:,2)+3*sdr(:,:,2))',(flip((mr(:,:,2)-3*sdr(:,:,2)),1))'],[6 6 6]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    subplot(326); plot(0:100,RG(:,:,3),'b');hold on;plot(0:100,RB(:,:,3),'r');title('Z')
    fill([0:100,flip(0:100,2)],[(mr(:,:,3)+sdr(:,:,3))',(flip((mr(:,:,3)-sdr(:,:,3)),1))'],[4 4 4]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(mr(:,:,3)+2*sdr(:,:,3))',(flip((mr(:,:,3)-2*sdr(:,:,3)),1))'],[5 5 5]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
    fill([0:100,flip(0:100,2)],[(mr(:,:,3)+3*sdr(:,:,3))',(flip((mr(:,:,3)-3*sdr(:,:,3)),1))'],[6 6 6]/8, 'EdgeColor', 'none', 'facealpha', 0.5);
end

%% MOVE THE CLEANED UP DATA OUT
L_out = L_data(:,L_good,:);
R_out = R_data(:,R_good,:);

end

function [ FFi_mod, FBi_mod, block_start, block_end ] = largest_block( FFi, FBi)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function LARGEST_BLOCK loads in  the foot forward/foot back indices 
% (FFi, FBi) and  outputs the longest chunk of continuous events from the
% initial index (FFi_mod, FBi_mod).
% Function also outputs the index of the start/end of the entire continuous
% block (block_start, block_end).
%
% This function is necessary as there are many reason why either a FF or FB
% event could be missed. An imbalanced index (more FFs or more FBs) causes
% many downstream problems with the code. 
%
%
%  INPUTS
%  --------
%  FFI/FBI (mat):   Frame index of foot-forward/foot-back events for
%                   entire length of trial
%
%  OUTPUTS
%  -------
%  FFI_MOD/FBI_MOD (mat):   Abbreviated index of foot-forward/foot-events 
%                           containing only the longest set of continuous 
%                           events.       
%
%  BLOCK_START/BLOCK_END (int): Frame index of when the largest block of 
%                               continuous events begins and ends

% Copyright (C) 2016-2023 Allan Brett and The Running Injury Clinic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Combine and sort the FF and FB
allsort = [FFi(:) zeros(length(FFi),1); FBi(:) ones(length(FBi),1)];
[~,inds] = sort(allsort(:,1));
allsort = allsort(inds,:);

% Remove trailing FF
while allsort(end,2) ~= 1
    allsort(end,:) = [];
end

allsort_bak = allsort;

% Confirm alternating status and remove trailing chunks that aren't
% ordered

i=0;
skip = 0;
longest_length =0;

%series of while loops will search for longest continuous chunk of data
%first while loop adds up the length of a continuous segments adding in the
%indicies that are skipped (because they contain the dicontinuity)
while (sum(longest_length)+skip < size(allsort_bak,1)) && size(allsort,1)>1
    
    idx_skip = 0;
    
    %points, in case of two 1s run below
    k = 1;
    
    %allsort must start with a 0
    while allsort(1,2) == 1
        allsort(1,:) = [];
        %skip is an overall counter for the main while loop in
        %conjuction with longest_length
        skip = skip+1;
        
        %idx_skip keeps track of when values are skipped for the
        %purpose of indexing
        idx_skip = idx_skip+ 1;
        
        %remove discontinuities occuring at the start of allsort
        while allsort(k,2)==allsort(k+1,2) && k+1<size(allsort,1)
            allsort(k:k+1,:) = [];
            skip = skip + 2;
            idx_skip = idx_skip + 2;
        end
    end
    
    k = 2;
    while k <= length(allsort) && mean(allsort(1:2:k,2)) == 0 && mean(allsort(2:2:k,2)) == 1
        k = k + 2;
    end
    
    i=i+1;
    
    %we don't want to use a possibly erroneous point in the data and we
    %must end the sequence on a 1, so when the dicontinuity occurs with two
    %1s in a row, we must roll back by 2
    if size(allsort,1) > k && k > 2
        if allsort(k-1,2)==1 && allsort(k-2,2) == 1
            allsort = allsort(1:k-4,:);
        else
            allsort = allsort(1:k-2,:);
        end
    end
    
    %for the special case where there are two discontinuities of 0s in a row
    if k == 2 && allsort(1,2) == 0
        allsort(1:2,:) = [];
        longest_length(i,1) = 0;
        
        %we want to index one passed the discontinuity
        if i == 1
            %if this occurs for the first index, only includes values
            %skipped
            index(1,1) = idx_skip;
            index(2,1) = idx_skip;
        else
            index(1,i) = index(2,i-1)+idx_skip+1;
            index(2,i) = index(2,i-1)+idx_skip+1;
        end
        skip = skip + 2;
        
    else
        
        %otherwise count as normal
        longest_length(i,1) = size(allsort,1);
        
        
        %create ordered index of where continuous chunks occur
        if i == 1
            index(1,1) = 1 + idx_skip;
            index(2,1) = longest_length(i,1)+idx_skip;
        else
            index(1,i) = index(2,i-1)+3+idx_skip;
            
            %Longest_length can only be 0 when
            %two discontinuities of 1s happen in a row, below accounts that
            %the index end needs to still progress by 1 (but longest_length
            %still needs to be 0 for the main counter)
            if longest_length(i,1) >0
                index(2,i) = index(1,i)+longest_length(i,1)-1;
            else
                index(2,i) = index(1,i)+longest_length(i,1);
            end
        end
        
        
        %reset allsort for next loop iteration to be passed the discontinuity
        allsort = allsort_bak(index(2,i)+3:end,:);
        
        
        %however we want to skip passed the discontinuity to the next footfall.
        %This entails skipping the discontinuity (for example if the
        %discontinuity is two FF, we skip over these two values
        
        skip = skip+ 2;
    end
    
end

%determine which index has the largest continuous block
[~,longest_index]  = max(diff(index));

%reorder allsort to contain only this block
allsort_longest = allsort_bak(index(1,longest_index):index(2,longest_index),:);
allsort = allsort_longest;

% Break back into components
FFi_mod = allsort(allsort(:,2) == 0,1);
FBi_mod = allsort(allsort(:,2) == 1,1);

%want to track frame numbers of when the block on continuous data starts
%and ends
block_start = FFi_mod(1,1);
block_end = FBi_mod(end,1);

end



function [output ] = oscillation(trsegment,L_FFi, L_FBi, R_FFi, R_FBi, plots, label)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION oscillation_run reads in a desired cluster and calculates its 
% oscillation in the veritcal plane.
%
%
%  INPUTS
%  --------
%  TRSEGMENT (struct):  Marker segment used to quantify oscillation.
%                       For gait we opt to use the average position of the
%                       3 markers of the PELVIC marker cluster.
%
%  L_FFI, L_FBI (mat):  Touchdown/toe-off  indices of the left feet
%
%  R_FFI, R_FBI (mat):  Touchdown/toe-off  indices of the right feet
%
%  HZ (int):            Sampling frequency.
%
%  PLOTS (bool):        Generate addition plotting for debugging if set
%                       equal to 1.
%
%  LABEL (str):         Gait type (walk/run) which serves to modify the
%                       window where the peak oscillation is calculated.
%
%  OUTPUTS
%  -------
%  OUTPUT (float):      Vertical oscillation (in mm) by stride separated by
%                       left and right stance
%
%
% Copyright (C) 2016-2023 Allan Brett and Running Injury Clinic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
build = 0;

%average out pelvic cluster points in the vertical plane
running_segment = trsegment(:,2);

%backup original curve for debugging
%running_segment_bak = running_segment;

A= [length(L_FFi), length(R_FFi), length(R_FBi), length(L_FBi)];

%% utlize moving window to minimize errors

%ignore first touchdown, too small of an initial window
%causes errors. Similarly, ignore the last touchdown

for k = 1:min(A)-1
    
    %define moving window, window from touchdown to
    %touchdown for each foot, add in 10 frame cushion
    
    if strcmp(label,'run')
        
        l_window = [L_FFi(k);L_FBi(k)+25];
        r_window = [R_FFi(k);R_FBi(k)+25] ;
        
    elseif strcmp(label,'walk')
        
        l_window = [L_FFi(k)-10;R_FFi(k)-5];
        r_window = [R_FFi(k)-10;L_FFi(k+1)-5] ;
        
    end
    
    
    %track peaks/troughs for each approximate stride
    left = running_segment(l_window(1,1):l_window(2,1));
    right = running_segment(r_window(1,1):r_window(2,1));
    
    [left_peak, left_peak_loc ] = findpeaks(left);
    [right_peak, right_peak_loc ] = findpeaks(right);
    [left_trough, left_trough_loc ] = findpeaks(-left);
    [right_trough, right_trough_loc ] = findpeaks(-right);
    
    %Occasionally an error in touchdown or toe-off index can lead to the
    %window being incorrect, this leads to multiple or no peaks. When this
    %occurs, return NaN and move on.
    if size(left_peak,1) ~= 1 || size(right_peak,1) ~= 1 || size(right_trough,1) ~=  1 || size(left_trough,1) ~= 1
        
        build = build + 1;
        peak_locs(build)=NaN;
        trough_locs(build) = NaN;
        peaks(build)=NaN;
        trough(build) = NaN;
        
        build = build+1;
        peak_locs(build) = NaN;
        trough_locs(build) = NaN;
        peaks(build)=NaN;
        trough(build) = NaN;
        
        
        
    else
        
        
        %check for inflection points instead of peaks
        
        for x = 1:length(left_peak)
            check_l_peak(x)  = find(left == left_peak(x));
            if left(check_l_peak(x)) < left(check_l_peak(x) + 1)
                left_peak(x) = 0;
                left_peak_loc(x) = 0;
            end
        end
        
        for x = 1:length(right_peak)
            check_r_peak(x)  = find(right == right_peak(x));
            if right(check_r_peak(x)) < right(check_r_peak(x) + 1)
                right_peak(x) = 0;
                right_peak_loc(x) = 0;
            end
        end
        
        for x = 1:length(left_trough)
            check_l_trough(x)  = find(-left == left_trough(x));
            if left(check_l_trough(x)) > left(check_l_trough(x) + 1)
                left_trough(x) = 0;
                left_trough_loc(x) = 0;
            end
        end
        
        for x = 1:length(right_trough)
            check_r_trough(x)  = find(-right == right_trough(x));
            if right(check_r_trough(x)) > right(check_r_trough(x) + 1)
                right_trough(x) = 0;
                right_trough_loc(x) = 0;
            end
        end
        
        %build peaks and troughs matrices by stacking right and lefts
        %this is necessary for confirming alternating status below
        build = build + 1;
        peak_locs(build)=left_peak_loc + L_FFi(k)-1;
        trough_locs(build) = left_trough_loc + L_FFi(k)-1;
        peaks(build)=left_peak;
        trough(build) = -left_trough;
        
        build = build+1;
        peak_locs(build) = right_peak_loc+R_FFi(k)-1;
        trough_locs(build) = right_trough_loc+R_FFi(k)-1;
        peaks(build)=right_peak;
        trough(build) = -right_trough;
        
    end
end

%set bottom and top edges for plotting
bot = min(running_segment);
top = max(running_segment);

%% calculate oscillation

if peak_locs(1) < trough_locs(1)
    peak_locs(1) = [];
    peaks(1) = [];
end

allsort = [peak_locs(:) zeros(length(peak_locs),1) peaks(:); trough_locs(:) ones(length(trough_locs),1) trough(:)];
[~,inds] = sort(allsort(:,1));
allsort = allsort(inds,:);

%remove trailing trough

while allsort(end,2) ~= 0
    allsort(end,:) = [];
end

k = 2;

%ensure alternating status

while k <= length(allsort) && mean(allsort(1:2:k,2)) == 1 && mean(allsort(2:2:k,2)) == 0
    k = k + 2;
end
allsort = allsort(1:k-2,:);

% Break back into components
peak_locs = allsort(allsort(:,2) == 0,1);
trough_locs = allsort(allsort(:,2) == 1,1);
peaks = allsort(allsort(:,2)==0,3);
trough = allsort(allsort(:,2)==1,3);

%debugging
if plots ~=0
    %plot the curves
    figure
    
    %include offset caused by chopping by first left footfall
    
    for i = 1:length(running_segment)
        if i == 1
            x_axis(i) = L_FFi(1);
        else
            x_axis(i) = x_axis(i-1) + 1;
        end
    end
    
    title('Vertical')
    
    plot (running_segment_bak)
    hold on
    plot(trough_locs,trough, 'r.')
    plot (peak_locs,peaks, 'g.')
    
    %plot touchdown/toeoffs as well
    for p=1:min(A)-1
        fill([L_FFi(p,1),L_FBi(p,1),L_FBi(p,1),L_FFi(p,1)],[bot, bot, top, top], ...
            [3 3 7]/8, 'EdgeColor','none','facealpha',0.2);
        fill([R_FFi(p,1),R_FBi(p,1),R_FBi(p,1),R_FFi(p,1)],[bot, bot, top, top], ...
            [7 3 3]/8, 'EdgeColor','none','facealpha',0.2);
        
    end
    hold off
end


%define oscillation as the distance between peak and trough

vert_oscillation =  peaks-trough;

%vert_oscillation always begin with stance for the left foot
left_stance = vert_oscillation(1:2:end,:);
right_stance =vert_oscillation(2:2:end,:);

%chop to shortest length
min_length = min(length(left_stance), length(right_stance));

left_stance = left_stance(1:min_length,1);
right_stance = right_stance(1:min_length,1);

for i = 1:min_length
    if i == 1
        stride_num(i,1)  = 1;
    else
        stride_num(i,1) = stride_num(i-1,1)+1;
    end
end

stride_num = peak_locs;

output = horzcat(stride_num(1:2:end), left_stance, stride_num(2:2:end), right_stance);

%% quality checks if there are more nans than numbers, something has gone wrong

left_check_nan = sum(isnan(left_stance));
left_check_num = sum(~isnan(left_stance));

right_check_nan = sum(isnan(right_stance));
right_check_num = sum(~isnan(right_stance));

if left_check_nan > left_check_num || right_check_nan > right_check_num
    
    error('Number of rejected strides for vertical oscillation too high')
    
end

clear allsort peak_locs peaks trough trough_locs balance oscillation_up oscillation_down left_ratio right_ratio R_FBi L_FBi L_FFi R_FFi

end

