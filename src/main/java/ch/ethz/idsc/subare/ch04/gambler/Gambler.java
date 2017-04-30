// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Range;
import ch.ethz.idsc.tensor.red.KroneckerDelta;
import ch.ethz.idsc.tensor.red.Min;

class Gambler implements StandardModel // , RewardInterface
{
  static final Scalar TERMINAL_W = RealScalar.of(100);
  // static final Scalar TERMINAL_L = ZeroScalar.get();
  final Tensor states = Range.of(0, 101);
  final Index statesIndex;
  final Scalar P_win;

  /** @param P_win probabilty of winning a coin toss */
  public Gambler(Scalar P_win) {
    statesIndex = Index.build(states);
    this.P_win = P_win;
  }

  @Override
  public Tensor actions(Tensor state) {
    Scalar stateS = state.Get();
    return Range.of(Min.of(stateS, TERMINAL_W.subtract(stateS)).number().intValue() + 1);
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    // return state.add(action);
    // non deterministic
    return null;
  }

  // @Override
  public Scalar reward(Tensor state, Tensor action) {
    return KroneckerDelta.of(state, TERMINAL_W);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    if (state.equals(TERMINAL_W))
      return RealScalar.ONE;
    // ---
    final Scalar stateS = state.Get();
    Scalar value = ZeroScalar.get();
    // headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action]
    {
      Scalar prob = P_win;
      Scalar next = stateS.add(action);
      Scalar reward = reward(next, null);
      Scalar res1; //
      res1 = prob.multiply(gvalues.Get(statesIndex.of(next)));
      // res1 = prob.multiply(reward.add(gvalues.Get(statesIndex.of(next))));
      value = value.add(res1);
    }
    {
      Scalar prob = RealScalar.ONE.subtract(P_win);
      Scalar next = stateS.add(action.negate());
      Scalar reward = reward(next, null);
      Scalar res2; // = prob.multiply(reward.add(gvalues.Get(statesIndex.of(next))));
      res2 = prob.multiply(gvalues.Get(statesIndex.of(next)));
      // res2 = prob.multiply(reward.add(gvalues.Get(statesIndex.of(next))));
      value = value.add(res2);
    }
    return value;
  }
}
