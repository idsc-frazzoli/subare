// code by jph
package ch.ethz.idsc.subare.ch02;

import java.util.Random;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;

public abstract class Agent {
  protected static final Random random = new Random();
  // ---
  private Scalar total = ZeroScalar.get();
  private int count = 0;
  private Tensor history = Tensors.empty();

  public abstract int takeAction();

  abstract void protected_feedReward(int a, Scalar value);

  public final void feedReward(int a, Scalar value) {
    ++count;
    total = total.plus(value);
    history.append(RealScalar.of(a));
    protected_feedReward(a, value);
  }

  public final Scalar getCount() {
    return RealScalar.of(count);
  }

  public Scalar getTotal() {
    return total;
  }

  public Tensor getHistory() {
    return history.unmodifiable();
  }

  public abstract String getDescription();

  public final String getAbsDesc() {
    String name = getClass().getSimpleName();
    name = name.substring(0, name.length() - 5); // drop "Agent"
    return String.format("%10s%15s", name, getDescription());
  }

  @Override
  public String toString() {
    return getAbsDesc();
  }
}
