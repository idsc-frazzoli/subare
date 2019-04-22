// code by jph
// adapted from code by Shangtong Zhang
package ch.ethz.idsc.subare.ch01.tic;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

class AllStates {
  public static final AllStates INSTANCE = new AllStates();
  // ---
  private final Map<Integer, State> allStates = new HashMap<>();
  private final Map<State, State> equivalence = new HashMap<>();

  private AllStates() {
    State newState = State.EMPTY;
    allStates.put(newState.hashCode, newState);
    getAllStatesImpl(newState, 1);
    for (State state : allStates.values())
      getEquivalent(state);
  }

  // recursive implementation
  void getAllStatesImpl(State currentState, int symbol) {
    for (int pos = 0; pos < 9; ++pos)
      if (currentState.data[pos] == 0) {
        State newState = currentState.nextState(pos, symbol);
        if (!allStates.containsKey(newState.hashCode)) {
          allStates.put(newState.hashCode, newState);
          if (!newState.isEnd())
            getAllStatesImpl(newState, -symbol);
        }
      }
  }

  public Set<State> getAll() {
    return new HashSet<>(allStates.values());
  }

  public State getRepresentative(State state) {
    // GlobalAssert.of(allStates.containsKey(state.hashCode));
    return allStates.get(state.hashCode);
  }

  public State getEquivalent(State state) {
    final State key = getRepresentative(state);
    if (!equivalence.containsKey(key)) {
      NavigableMap<Integer, State> subset = new TreeMap<>();
      {
        for (int i = 0; i < 4; ++i) {
          state = state.getRotated();
          subset.put(state.hashCode, state);
        }
        state = state.getMirrored();
        for (int i = 0; i < 4; ++i) {
          state = state.getRotated();
          subset.put(state.hashCode, state);
        }
      }
      equivalence.put(key, getRepresentative(subset.firstEntry().getValue()));
    }
    return equivalence.get(key);
  }

  public Set<State> getEquivalenceSet() {
    return new HashSet<>(equivalence.values());
  }

  public State getFromHash(int key) {
    return allStates.get(key);
  }

  public static void main(String[] args) {
    for (State s : INSTANCE.allStates.values()) {
      if (s.isEnd())
        System.out.println(s);
    }
    System.out.println(INSTANCE.allStates.size());
    System.out.println(INSTANCE.getEquivalenceSet().size());
    // instance.allStates
    // System.out.println(instance.getRepresentative(State.empty));
    // System.out.println(instance.getRepresentative(new State(new int[9])));
  }
}
