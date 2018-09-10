// code by jph and fluric
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;

class UcbPolicyBuilder {
  private final DiscreteModel discreteModel;
  private final QsaInterface qsa;
  private final StateActionCounter sac;
  final Map<Tensor, Index> map = new HashMap<>();
  final Map<Tensor, Integer> sizes = new HashMap<>();

  public UcbPolicyBuilder(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.sac = sac;
  }

  // this simplicity may be the reason why q(s,a) is preferred over v(s)
  public void append(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().parallel().map(action -> UcbUtils.getUpperConfidenceBound(state, action, qsa, sac, discreteModel)));
    FairArgMax fairArgMax = FairArgMax.of(va);
    Tensor feasible = Tensor.of(fairArgMax.options().stream().map(actions::get));
    // Tensor feasible = Extract.of(actions, fairArgMax.options());
    map.put(state, Index.build(feasible));
    sizes.put(state, actions.length());
  }
}
