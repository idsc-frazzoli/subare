// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.red.Max;
import ch.ethz.idsc.tensor.red.Norm;

/** measure to compare performance between optimal state-action value function and learned q-function */
public enum Loss {
  ;
  /** @param discreteModel
   * @param ref ground truth
   * @param qsa
   * @return non-negative number which should be subtracted from the optimal gains */
  public static Scalar accumulation(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    return Norm._1.of(perState(discreteModel, ref, qsa).values());
  }

  /** @param discreteModel
   * @param ref ground truth
   * @param qsa
   * @return map that assigns each state a non-negative number which is the deficiency
   * of the evaluation of the state from the optimal evaluation */
  public static DiscreteVs perState(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    return DiscreteUtils.createVs(discreteModel, asQsa(discreteModel, ref, qsa), Scalar::add);
  }

  /** @param discreteModel
   * @param ref ground truth
   * @param qsa
   * @return map that assigns each state a non-negative number which is the deficiency
   * of the evaluation of the state from the optimal evaluation */
  public static DiscreteQsa asQsa(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    DiscreteQsa loss = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states()) {
      Tensor actions = discreteModel.actions(state);
      Scalar max = actions.flatten(0) //
          .map(action -> ref.value(state, action)) //
          .reduce(Max::of).get();
      // ---
      FairArgMax fairArgMax = FairArgMax.of(Tensor.of(actions.flatten(0).map(action -> qsa.value(state, action))));
      Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount());
      for (int index : fairArgMax.options()) {
        Tensor action = actions.get(index);
        Scalar delta = max.subtract(ref.value(state, action));
        if (Scalars.lessThan(delta, RealScalar.ZERO))
          throw TensorRuntimeException.of(delta);
        loss.assign(state, action, delta.multiply(weight));
      }
    }
    return loss;
  }
}
