%Knowledge base
% 7.1

% a)

:- op(800, xfy, and).
:- op(810, xfy, or).
:- op(950, xfx, then).
:- op(850, fx, if).
:- op(750, xfx, with).

% R1
if motor=nao and bateria=ma then problema=bateria with fc=1.
% R2
if luz=fraca then bateria=ma with fc=0.8.
% R3
if radio=fraco then bateria=ma with fc=0.8.
% R4
if luz=boa and radio=bom then bateria=boa with fc=0.8.
% R5
if motor=sim and cheiro_gas=sim then problema=encharcado with fc=0.8.
% R6
if motor=nao and bateria=boa and indicador_gas=vazio then problema=sem_gasolina with fc=0.9.
% R7
if motor=nao and bateria=boa and indicador_gas=baixo then problema=sem_gasolina with fc=0.3.
% R8
if motor=nao and cheiro_gas=nao and ruido_motor=nao_ritmado and bateria=boa then problema=motor_gripado with fc=0.7.
% R9
if motor=nao and cheiro_gas=nao and bateria=boa then problema=carburador_entupido with fc=0.9.
% R10
if motor=nao and bateria=boa then problema=velas_estragadas with fc=0.8.

% b)

:- dynamic(fact/3).
% fact(A, V, FC).

% c)

questionable(motor, 'O motor funciona?', [sim,nao]).
questionable(luz, 'Como estao as luzes?', [fraca,razoavel,boa]).
questionable(radio, 'Como esta o radio?', [fraco,razoavel,bom]).
questionable(cheiro_gas, 'Sente cheiro a gasolina?', [sim,nao]).
questionable(indicador_gas, 'Como esta o indicador de gasolina?', [vazio,baixo,meio,cheio]).
questionable(ruido_motor, 'Que ruido faz o motor?', [ritmado,nao_ritmado]).



%Implementation

%Already in database
check(A,V,FC) :-
    fact(A,V,FC), !.

%There is a fact in the database but with different values
check(A,V,_) :-
    fact(A,V2,_),
    V2 \= V, !, fail.

%No fact in database, ask the user if possible
check(A,V,FC) :-
    questionable(A, Question, LR),
    repeat,
    write(Question:LR), read(Response),
    member(Response, LR),
    write('Certainty [0-1]?'), read(FC),
    assert(fact(A,Response,FC)),
    !, Response = V.

%No fact in database, can't ask user. Check if value can be calculated.
check(A,V,FC) :-
    deduct(A,V,FC).


deduct(A,V,FC) :-
    if Premise then A=V with FC=FCRule,
    prove(Premise, FCPremise),
    FCNew is FCPremise * FCRule,
    update(A,V,FCNew),
    fail.

deduct(A,V,FC) :-
    fact(A,V,FC).


prove(A=V, FC) :-
    check(A,V,FC).

prove(A=V and Ps, FC) :-
    check(A,V,FC1),
    check(Ps, FC2),
    FC is min(FC1, FC2).

prove(A=V or Ps, FC) :-
    check(A,V,FC1),
    check(Ps, FC2),
    FC is max(FC1, FC2).


update(A,V,FCNew) :-
    ((fact(A,V,FCOld), !,
     retract(fact(A,V,FCOld)),
     FC is FCNew + FCOld * (1 - FCNew));
    (FC is FCNew)),
    assert(fact(A,V,FC)).


start :- 
    retractall(fact(_,_,_)),
    check(problem,V,FC),
    write(problem = V : FC).
