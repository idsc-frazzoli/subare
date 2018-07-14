// code by jph
// adapted from code by Shangtong Zhang
package ch.ethz.idsc.subare.ch01.tic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.TreeMap;

import ch.ethz.idsc.subare.util.Coinflip;
import ch.ethz.idsc.tensor.RealScalar;

class Agent {
  double stepSize;
  // double exploreRate;
  private Coinflip coinFlip;
  List<State> states = new ArrayList<>();
  final int symbol;
  Estimation estimation = new Estimation();
  protected State currentState;

  public Agent(int symbol) {
    this.symbol = symbol;
  }

  public void setRates(double stepSize, double exploreRate) {
    this.stepSize = stepSize;
    // this.exploreRate = exploreRate;
    coinFlip = Coinflip.of(RealScalar.of(exploreRate));
  }

  // accept a state
  final void feedState(State state) {
    currentState = AllStates.instance.getRepresentative(state);
    states.add(currentState);
  }

  // update estimation according to reward
  final void feedReward(double reward) {
    Collections.reverse(states);
    for (State latestState : states) {
      double value = estimation.get(latestState) //
          + stepSize * (reward - estimation.get(latestState));
      estimation.put(latestState, value);
      reward = value;
    }
  }

  // a policy is a mapping from perceived states of the environment to actions to
  // be taken when in those states.
  public Action takeAction() {
    Map<Integer, State> nextStates = new HashMap<>();
    for (int pos = 0; pos < 9; ++pos)
      if (currentState.data[pos] == 0) {
        State newState = currentState.nextState(pos, symbol);
        nextStates.put(pos, newState);
      }
    // ---
    // exploration
    // parameter p denotes the probability of the outcome 1
    if (coinFlip.tossHead()) {
      List<Entry<Integer, State>> list = new ArrayList<>(nextStates.entrySet());
      Collections.shuffle(list);
      return new Action(list.get(0).getKey(), symbol);
    }
    // ---
    NavigableMap<Double, List<Integer>> est = new TreeMap<>();
    for (Entry<Integer, State> entry : nextStates.entrySet()) {
      double key = estimation.get(entry.getValue());
      if (!est.containsKey(key))
        est.put(key, new ArrayList<>());
      est.get(key).add(entry.getKey());
    }
    List<Integer> list = est.lastEntry().getValue();
    Collections.shuffle(list);
    return new Action(list.get(0), symbol);
  }

  public void reset() {
    states.clear();
  }

  void savePolicy() {
    Estimation.save(estimation, symbol);
  }

  void loadPolicy() {
    estimation = Estimation.load(symbol);
  }
}
