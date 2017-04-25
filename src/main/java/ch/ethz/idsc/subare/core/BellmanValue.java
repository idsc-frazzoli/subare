// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Norm;

public class BellmanValue {
  public static Tensor iteration(StandardModel standardModel, Index statesIndex, Index actionsIndex, Scalar discount, Scalar treshold) {
    final int n = statesIndex.size();
    Tensor v = Array.zeros(n); // TODO initial value
    while (true) {
      Tensor v_new = Array.zeros(n);
      for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
        Tensor state = statesIndex.get(stateI);
        for (int actionI = 0; actionI < actionsIndex.size(); ++actionI) {
          Tensor action = actionsIndex.get(actionI);
          // ---
          Tensor next = standardModel.move(state, action);
          Scalar policy = standardModel.policy(state, action);
          Scalar reward = standardModel.reward(state, action);
          // # bellman equation
          // newWorld[i, j] += actionProb[i][j][action] * (actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])
          int nextI = statesIndex.indexOf(next);
          Scalar delta = policy.multiply(reward.add(discount.multiply(v.Get(nextI))));
          v_new.set(s -> s.add(delta), stateI);
        }
      }
      if (Scalars.lessThan(Norm._1.of(v_new.subtract(v)), treshold))
        break;
      v = v_new.unmodifiable();
    }
    return v;
  }
}
