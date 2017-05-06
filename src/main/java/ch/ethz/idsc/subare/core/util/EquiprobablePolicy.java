// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** the term "equiprobable" appears in Exercise 4.1 */
public class EquiprobablePolicy implements PolicyInterface {
  final DiscreteModel discreteModel;
  final Map<Tensor, Scalar> map = new ConcurrentHashMap<>();

  public EquiprobablePolicy(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    if (!map.containsKey(state)) {
      Tensor actions = discreteModel.actions(state);
      Index actionIndex = Index.build(actions);
      if (!actionIndex.containsKey(action))
        throw new RuntimeException("action invalid " + action);
      int den = actions.length();
      if (den == 0)
        throw new RuntimeException();
      map.put(state, RationalScalar.of(1, den));
    }
    return map.get(state);
  }
}
