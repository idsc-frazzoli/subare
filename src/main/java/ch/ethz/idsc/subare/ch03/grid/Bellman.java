// code by jph
package ch.ethz.idsc.subare.ch03.grid;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

class Bellman {
  public static void main(String[] args) {
    GridWorld gw = new GridWorld();
    Index actionsIndex = Index.of(gw.actions);
    Index statesIndex = Index.of(gw.states);
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      for (int actionI = 0; actionI < actionsIndex.size(); ++actionI) {
        Tensor action = actionsIndex.get(actionI);
        Scalar reward = gw.reward(state, action);
        Tensor next = gw.move(state, action);
        int nextI = statesIndex.indexOf(next);
        System.out.println(next + " " + reward);
      }
    }
  }
}
