// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.function.BinaryOperator;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Max;

public enum DiscreteUtils {
  ;
  /** @param discreteModel
   * @return index for state-action */
  public static Index build(DiscreteModel discreteModel, Tensor states) {
    Tensor tensor = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : discreteModel.actions(state))
        tensor.append(Tensors.of(state, action));
    return Index.build(tensor);
  }

  // ---
  /** compute state value function v(s) based on given action-value function q(s,a)
   * 
   * @param discreteModel
   * @param qsa
   * @return state values */
  public static DiscreteVs createVs(DiscreteModel discreteModel, QsaInterface qsa) {
    // return DiscreteVs.build(discreteModel, //
    // Tensor.of(discreteModel.states().flatten(0) //
    // .map(state -> discreteModel.actions(state).flatten(0) //
    // .map(action -> qsa.value(state, action)) //
    // .reduce(Max::of).get()))); // <- assumes greedy policy
    return createVs(discreteModel, qsa, Max::of);
  }

  public static DiscreteVs createVs(DiscreteModel discreteModel, QsaInterface qsa, BinaryOperator<Scalar> op) {
    return DiscreteVs.build(discreteModel, //
        Tensor.of(discreteModel.states().flatten(0) //
            .map(state -> discreteModel.actions(state).flatten(0) //
                .map(action -> qsa.value(state, action)) //
                .reduce(op).get()))); // <- assumes greedy policy
  }

  /** @param discreteModel
   * @param qsa
   * @param state
   * @return action with max value, or random action among the actions with max value */
  public static Tensor fairBestAction(DiscreteModel discreteModel, QsaInterface qsa, Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor qvalues = Tensor.of(actions.flatten(0).map(action -> qsa.value(state, action)));
    FairArgMax fairArgMax = FairArgMax.of(qvalues);
    return actions.get(fairArgMax.nextRandomIndex());
  }
}
