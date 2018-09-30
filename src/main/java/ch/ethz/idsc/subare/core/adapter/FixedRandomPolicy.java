// code by jph
package ch.ethz.idsc.subare.core.adapter;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionModel;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.pdf.Distribution;

public class FixedRandomPolicy implements Policy {
  private static final Random RANDOM = new Random();
  // ---
  private final Set<Tensor> set = new HashSet<>();

  public FixedRandomPolicy(StateActionModel stateActionModel) {
    for (Tensor state : stateActionModel.states()) {
      Tensor actions = stateActionModel.actions(state);
      set.add(Tensors.of(state, actions.get(RANDOM.nextInt(actions.length()))));
    }
  }

  @Override // from Policy
  public final Scalar probability(Tensor state, Tensor action) {
    // TODO V061
    return set.contains(Tensors.of(state, action)) ? RealScalar.ONE : RealScalar.ZERO;
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    throw new UnsupportedOperationException();
  }
}
