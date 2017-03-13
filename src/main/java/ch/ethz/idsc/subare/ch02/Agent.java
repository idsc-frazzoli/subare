// code by jph
package ch.ethz.idsc.subare.ch02;

import java.util.Random;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.ZeroScalar;

public abstract class Agent {
  protected static final Random random = new Random();
  // ---
  private RealScalar total = ZeroScalar.get();
  private int count = 0;

  public abstract int takeAction();

  abstract void protected_feedReward(int a, RealScalar value);

  public final void feedReward(int a, RealScalar value) {
    ++count;
    total = (RealScalar) total.plus(value);
    protected_feedReward(a, value);
  }

  public int getCount() {
    return count;
  }

  public RealScalar getTotal() {
    return total;
  }

  public abstract String getDescription();

  public final String getAbsDesc() {
    String name = getClass().getSimpleName();
    name = name.substring(0, name.length() - 5);
    return String.format("%10s%15s", name, getDescription());
  }

  @Override
  public String toString() {
    return getAbsDesc();// + " \t" + Round.of(getTotal());
  }
}
