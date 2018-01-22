// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;

class EGreedyPolicyBuilder {
  private final DiscreteModel discreteModel;
  private final QsaInterface qsa;
  final Map<Tensor, Index> map = new HashMap<>();
  final Map<Tensor, Integer> sizes = new HashMap<>();

  public EGreedyPolicyBuilder(DiscreteModel discreteModel, QsaInterface qsa) {
    this.discreteModel = discreteModel;
    this.qsa = qsa;
  }

  // this simplicity may be the reason why q(s,a) is preferred over v(s)
  public void append(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().map(action -> qsa.value(state, action)));
    FairArgMax fairArgMax = FairArgMax.of(va);
    Tensor feasible = Tensor.of(fairArgMax.options().stream().map(actions::get));
    // Tensor feasible = Extract.of(actions, fairArgMax.options());
    map.put(state, Index.build(feasible));
    sizes.put(state, actions.length());
  }
}
