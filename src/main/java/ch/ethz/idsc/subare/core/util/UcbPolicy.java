// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** upper confidence bound is greedy except that it encourages
 * exploration if an action has not been encountered often relative to other actions */
public class UcbPolicy extends EGreedyPolicy {
  public static Policy bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac, Scalar epsilon) {
    UcbPolicyBuilder builder = new UcbPolicyBuilder(discreteModel, qsa, sac);
    discreteModel.states().forEach(builder::append);
    return new EGreedyPolicy(builder.map, epsilon, builder.sizes);
  }

  public static Policy bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac, Scalar epsilon, Tensor state) {
    UcbPolicyBuilder builder = new UcbPolicyBuilder(discreteModel, qsa, sac);
    builder.append(state);
    return new EGreedyPolicy(builder.map, epsilon, builder.sizes);
  }

  // ---
  protected UcbPolicy(Map<Tensor, Index> map, Scalar epsilon, Map<Tensor, Integer> sizes) {
    super(map, epsilon, sizes);
  }
}
