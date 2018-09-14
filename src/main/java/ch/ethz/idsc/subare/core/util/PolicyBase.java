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

public abstract class PolicyBase implements Policy, QsaInterfaceSupplier, StateActionCounterSupplier {
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
    // might be inefficient or even stale information
    this.qsa = DiscreteUtils.getQsaFromVs(standardModel, vs);
  }

  public final void setQsa(QsaInterface qsa) {
    this.qsa = qsa;
  }

  @Override // from QsaInterfaceSupplier
  public final QsaInterface qsaInterface() {
    return qsa;
  }

  public final void setSac(StateActionCounter sac) {
    this.sac = sac;
  }

  @Override // from StateActionCounterSupplier
  public final StateActionCounter sac() {
    return sac;
  }

  /** @param state
   * @return vector of actions that are equally optimal */
  public abstract Tensor getBestActions(Tensor state);

  public abstract PolicyBase copyOf(PolicyBase policyBase);
}
