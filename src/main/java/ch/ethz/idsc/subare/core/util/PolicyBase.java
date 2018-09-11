// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

abstract class PolicyBase implements Policy {
  protected final Map<Tensor, Index> stateToBestActions = new HashMap<>();
  protected final Map<Tensor, Integer> stateToActionSize = new HashMap<>();

  protected PolicyBase(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac, Tensor states) {
    for (Tensor state : states) {
      appendToMaps(discreteModel, qsa, sac, state);
    }
  }

  protected PolicyBase() {
  }

  abstract protected void appendToMaps(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac, Tensor state);

  @Override // from Policy
  abstract public Scalar probability(Tensor state, Tensor action);

  /** useful for export to Mathematica
   * 
   * @param states
   * @return list of actions optimal for */
  public Tensor flatten(Tensor states) {
    Tensor result = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : stateToBestActions.get(state).keys())
        result.append(Tensors.of(state, action));
    return result;
  }

  /** print overview of possible actions for given states in console
   * 
   * @param states */
  public void print(Tensor states) {
    System.out.println("greedy:");
    for (Tensor state : states)
      System.out.println(state + " -> " + stateToBestActions.get(state).keys());
  }

  public Tensor getBestActions(Tensor state) {
    return stateToBestActions.get(state).keys();
  }
}
