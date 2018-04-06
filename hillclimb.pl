:- use_module(library(lists)).
:- use_module(library(between)).

slots(4).
disciplinas(12).
disciplina(1,[1,2,3,4,5]).
disciplina(2,[6,7,8,9]).
disciplina(3,[10,11,12]).
disciplina(4,[1,2,3,4]).
disciplina(5,[5,6,7,8]).
disciplina(6,[9,10,11,12]).
disciplina(7,[1,2,3,5]).
disciplina(8,[6,7,8]).
disciplina(9,[4,9,10,11,12]).
disciplina(10,[1,2,4,5]).
disciplina(11,[3,6,7,8]).
disciplina(12,[9,10,11,12]).

incompat(D1, D2, NA) :-
  disciplina(D1, LA1),
  disciplina(D2, LA2),
  findall(A, (member(A,LA1), member(A,LA2)), LA12),
  length(LA12, NA).

f_aval(L,V) :-
  findall(N, (nth1(D1,L,Slot), nth1(D2,L,Slot), D1 < D2, incompat(D1,D2,N)), LIncompat),
  sumlist(LIncompat, V).

best(V1,V2) :- V1 < V2.

hillclimb(S, LocalOpt) :-
  f_aval(S,V1),
  neighbour(S,S2),
  f_aval(S2,V2),
  best(V2,V1), %MODIFICAR ISTO PARA TER SIMULATED ANEALING
  !, write(S2:V2), nl,
  hillclimb(S2, LocalOpt).
hillclimb(S,S).

neighbour(L, L2) :-
  slots(NSlots),
  nth1(D,L,Slot),
  between(1, NSlots, NovoSlot),
  NovoSlot \= Slot,
  D1 is D-1, length(Prefix, D1),
  append(Prefix, [Slot|Suffix], L),
  append(Prefix, [NovoSlot|Suffix], L2).
