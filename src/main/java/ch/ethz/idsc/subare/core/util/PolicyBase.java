// code by jph, fluric
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.QsaInterfaceSupplier;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

abstract public class PolicyBase implements Policy, QsaInterfaceSupplier, StateActionCounterSupplier {
  protected final DiscreteModel discreteModel;
  // ---
  protected StateActionCounter sac;
  protected QsaInterface qsa;

  protected PolicyBase(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    this.discreteModel = discreteModel;
    this.sac = sac;
    this.qsa = qsa;
  }

  protected PolicyBase(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    this.discreteModel = standardModel;
    this.sac = sac;
    this.qsa = getQsaFromVs(standardModel, vs, standardModel.states()); // might be inefficient or even stale information
  }

  private QsaInterface getQsaFromVs(StandardModel standardModel, VsInterface vs, Tensor states) {
    ActionValueAdapter actionValueAdapter = new ActionValueAdapter(standardModel);
    DiscreteQsa qsa = DiscreteQsa.build(standardModel);
    for (Tensor state : states) {
      for (Tensor action : standardModel.actions(state)) {
        qsa.assign(state, action, actionValueAdapter.qsa(state, action, vs));
      }
    }
    return qsa;
  }

  abstract protected Tensor getBestActions(DiscreteModel discreteModel, Tensor state);

  /** useful for export to Mathematica
   * 
   * @param states
   * @return list of actions optimal for */
  public Tensor flatten(Tensor states) {
    Tensor result = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : getBestActions(discreteModel, state))
        result.append(Tensors.of(state, action));
    return result;
  }

  /** print overview of possible best actions for given states
   * 
   * @param states */
  public void print(Tensor states) {
    System.out.println("greedy:");
    for (Tensor state : states)
      System.out.println(state + " -> " + getBestActions(discreteModel, state));
  }

  /** @param state
   * @return the best actions put in a {@link Tensor} */
  public Tensor getBestActions(Tensor state) {
    return getBestActions(discreteModel, state);
  }

  public void setQsa(QsaInterface qsa) {
    this.qsa = qsa;
  }

  @Override // from QsaInterfaceSupplier
  public QsaInterface qsaInterface() {
    return qsa;
  }

  public void setSac(StateActionCounter sac) {
    this.sac = sac;
  }

  @Override // from StateActionCounterSupplier
  public StateActionCounter sac() {
    return sac;
  }

  abstract public PolicyBase copyOf(PolicyBase policyBase);
}
