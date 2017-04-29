// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Norm;

/**
 * 
 */
public class ValueFunctions {
  /** iterative policy evaluation (4.5)
   * see box on p.81
   * 
   * under the assumption that p(s',r | s,a) \in {0,1}, i.e.
   * p(s',r | s,a) does not require special treatment or looping...
   * 
   * initial values are set to zeros
   * 
   * @param standardModel
   * @param policyInterface
   * @param statesIndex
   * @param gamma discount
   * @param threshold
   * @return */
  public static Tensor bellmanIteration( //
      StandardModel standardModel, PolicyInterface policyInterface, //
      Index statesIndex, Scalar gamma, Scalar threshold) {
    final int n = statesIndex.size();
    Tensor v = Array.zeros(n);
    while (true) {
      Tensor v_new = Array.zeros(n); // values are added to 0's during iteration
      Tensor gvalues = v.multiply(gamma);
      for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
        final Tensor state = statesIndex.get(stateI);
        // general bellman equation:
        // v_pi(s) == Sum_a pi(a|s) * Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
        // simplifies here to
        // v_pi(s) == Sum_a pi(a|s) * (r + gamma * v_pi(s'))
        for (Tensor action : standardModel.actions()) {
          Scalar policy = policyInterface.policy(state, action);
          Scalar delta = policy.multiply(standardModel.qsa(state, action, gvalues));
          v_new.set(s -> s.add(delta), stateI);
        }
      }
      if (Scalars.lessThan(Norm._1.of(v_new.subtract(v)), threshold))
        break;
      v = v_new.unmodifiable();
    }
    return v;
  }

  /** implementation of (3.17) on p.69
   * 
   * approximately equivalent to iterating with {@link GreedyPolicy}
   * 
   * initial values are set to zeros
   * 
   * @param standardModel
   * @param statesIndex
   * @param gamma discount
   * @param threshold
   * @return */
  public static Tensor bellmanIterationMax( //
      StandardModel standardModel, //
      Index statesIndex, Scalar gamma, Scalar threshold) {
    final int n = statesIndex.size();
    Tensor v = Array.zeros(n);
    while (true) {
      Tensor v_new = Array.zeros(n); // values are added to 0's during iteration
      Tensor gvalues = v.multiply(gamma);
      for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
        final Tensor state = statesIndex.get(stateI);
        // bellman optimality equation:
        // v_*(s) == max_a Sum_{s',r} p(s',r | s,a) * (r + gamma * v_*(s'))
        // simplifies here to
        // v_*(s) == max_a (r + gamma * v_*(s'))
        Tensor va = Tensors.empty();
        // TODO can reduce code by streaming va
        for (Tensor action : standardModel.actions())
          va.append(standardModel.qsa(state, action, gvalues));
        v_new.set(va.flatten(-1).reduce(Max::of).get(), stateI);
      }
      if (Scalars.lessThan(Norm._1.of(v_new.subtract(v)), threshold))
        break;
      v = v_new.unmodifiable();
    }
    return v;
  }
}
