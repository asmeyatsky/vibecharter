# Architectural Principles and Code Generation Standards

## Overview

This skill enforces strict architectural principles and code generation standards to ensure all code follows enterprise-grade patterns. When this skill is active, Claude must generate code that adheres to these principles without exception.

## 🏗️ Four Core Architectural Principles

### 1. Separation of Concerns (SoC)
- **Principle**: Each module/component should have a single, well-defined responsibility
- **Implementation**:
  - Separate data access, business logic, and presentation layers
  - Use dependency injection to manage dependencies
  - Create focused, single-purpose classes and functions
  - Avoid mixing concerns (e.g., UI logic in data access layer)

### 2. Domain-Driven Design (DDD)
- **Principle**: Software design should reflect the business domain
- **Implementation**:
  - Create rich domain models that encapsulate business rules
  - Use ubiquitous language from the business domain
  - Implement aggregates, entities, and value objects
  - Define clear bounded contexts
  - Use domain events for cross-boundary communication

### 3. Clean/Hexagonal Architecture
- **Principle**: Business logic should be independent of frameworks and infrastructure
- **Implementation**:
  ```
  Domain Layer (Core)
  ├── Entities
  ├── Value Objects
  ├── Domain Services
  └── Repository Interfaces
  
  Application Layer
  ├── Use Cases
  ├── DTOs
  └── Application Services
  
  Infrastructure Layer
  ├── Database Implementation
  ├── External Service Adapters
  └── Framework-specific Code
  
  Presentation Layer
  ├── Controllers/Handlers
  ├── View Models
  └── UI Components
  ```

### 4. High Cohesion, Low Coupling
- **Principle**: Related functionality should be grouped together, dependencies minimized
- **Implementation**:
  - Group related functionality in modules
  - Use interfaces to define contracts
  - Minimize dependencies between modules
  - Favor composition over inheritance

## 🛡️ Five Non-Negotiable Rules

### Rule 1: Zero Business Logic in Infrastructure Components
```python
# ❌ WRONG - Business logic in repository
class UserRepository:
    def get_active_premium_users(self):
        users = self.db.query("SELECT * FROM users")
        # Business logic should NOT be here!
        return [u for u in users if u.is_active and u.subscription == 'premium']

# ✅ CORRECT - Business logic in domain service
class UserRepository:
    def get_all_users(self):
        return self.db.query("SELECT * FROM users")

class UserDomainService:
    def get_active_premium_users(self, repository):
        users = repository.get_all_users()
        return [u for u in users if self._is_active_premium(u)]
    
    def _is_active_premium(self, user):
        return user.is_active and user.subscription == 'premium'
```

### Rule 2: Interface-First Development (Ports and Adapters)
```python
# Define ports (interfaces) first
from abc import ABC, abstractmethod

class PaymentGatewayPort(ABC):
    @abstractmethod
    def process_payment(self, amount: float, currency: str) -> PaymentResult:
        pass

class NotificationPort(ABC):
    @abstractmethod
    def send_notification(self, recipient: str, message: str) -> bool:
        pass

# Then implement adapters
class StripePaymentAdapter(PaymentGatewayPort):
    def process_payment(self, amount: float, currency: str) -> PaymentResult:
        # Stripe-specific implementation
        pass

class EmailNotificationAdapter(NotificationPort):
    def send_notification(self, recipient: str, message: str) -> bool:
        # Email-specific implementation
        pass
```

### Rule 3: Immutable Domain Models
```python
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class Order:
    id: str
    customer_id: str
    total_amount: float
    status: str
    created_at: datetime
    
    def mark_as_paid(self) -> 'Order':
        # Return new instance instead of mutating
        return replace(self, status='PAID')
    
    def apply_discount(self, percentage: float) -> 'Order':
        new_amount = self.total_amount * (1 - percentage / 100)
        return replace(self, total_amount=new_amount)
```

### Rule 4: Mandatory Testing Coverage
```python
# Every generated component must include tests

# Domain model test
class TestOrder:
    def test_mark_as_paid_creates_new_instance(self):
        order = Order(id="1", customer_id="C1", total_amount=100.0, 
                     status="PENDING", created_at=datetime.now())
        paid_order = order.mark_as_paid()
        
        assert paid_order.status == "PAID"
        assert order.status == "PENDING"  # Original unchanged
        assert paid_order is not order  # Different instances

# Use case test
class TestCreateOrderUseCase:
    def test_create_order_with_valid_data(self):
        # Arrange
        repository = Mock(OrderRepositoryPort)
        use_case = CreateOrderUseCase(repository)
        
        # Act
        result = use_case.execute(customer_id="C1", items=[...])
        
        # Assert
        repository.save.assert_called_once()
        assert result.is_success
```

### Rule 5: Documentation of Architectural Intent
```python
"""
Order Processing Module

Architectural Intent:
- This module handles order lifecycle management following DDD principles
- Order aggregate is the consistency boundary
- All state changes go through domain methods to ensure invariants
- External payment processing is handled via ports/adapters pattern
- Events are published for other bounded contexts to react

Key Design Decisions:
1. Orders are immutable to prevent accidental state corruption
2. Payment processing is abstracted behind PaymentGatewayPort
3. Order status transitions are validated in domain model
4. Complex pricing logic is delegated to PricingDomainService
"""

class OrderAggregate:
    """
    Order Aggregate Root
    
    Invariants:
    - Order total must be positive
    - Status transitions must follow defined state machine
    - Cancelled orders cannot be modified
    """
    pass
```

## 📋 Implementation Checklist

When generating code, Claude must verify:

### Layer Separation
- [ ] Domain layer has NO dependencies on infrastructure
- [ ] Application layer depends only on domain layer
- [ ] Infrastructure layer implements interfaces defined in domain
- [ ] Presentation layer only interacts with application layer

### Interface Design
- [ ] All external dependencies have interface definitions
- [ ] Interfaces are defined in the domain/application layer
- [ ] Concrete implementations are in infrastructure layer
- [ ] Dependency injection is used to wire implementations

### Domain Modeling
- [ ] Domain models are immutable where possible
- [ ] Business rules are encapsulated in domain objects
- [ ] Value objects are used for concepts without identity
- [ ] Aggregates maintain consistency boundaries

### Testing Requirements
- [ ] Unit tests for all domain logic
- [ ] Integration tests for infrastructure adapters
- [ ] Use case tests with mocked dependencies
- [ ] Test coverage meets minimum 80% threshold

### Documentation Standards
- [ ] Each module has architectural intent documented
- [ ] Key design decisions are recorded
- [ ] Domain concepts are explained
- [ ] Integration points are clearly marked

## 🎯 Code Generation Guidelines

### When Creating New Components

1. **Start with the Domain**
   - Define entities and value objects first
   - Identify aggregate boundaries
   - Document invariants and business rules

2. **Define Interfaces**
   - Create port interfaces for external dependencies
   - Design application service interfaces
   - Keep interfaces focused and cohesive

3. **Implement Use Cases**
   - One use case per class
   - Orchestrate domain objects
   - Handle cross-cutting concerns

4. **Add Infrastructure**
   - Implement adapters for ports
   - Configure dependency injection
   - Add persistence mappings

5. **Create Tests**
   - Test domain logic independently
   - Mock external dependencies
   - Verify architectural boundaries

### Example Project Structure
```
project/
├── domain/
│   ├── entities/
│   │   ├── order.py
│   │   └── customer.py
│   ├── value_objects/
│   │   ├── money.py
│   │   └── address.py
│   ├── services/
│   │   └── pricing_service.py
│   └── ports/
│       ├── repository_ports.py
│       └── external_service_ports.py
├── application/
│   ├── use_cases/
│   │   ├── create_order.py
│   │   └── process_payment.py
│   └── dtos/
│       └── order_dto.py
├── infrastructure/
│   ├── repositories/
│   │   └── order_repository.py
│   ├── adapters/
│   │   ├── payment_adapter.py
│   │   └── notification_adapter.py
│   └── config/
│       └── dependency_injection.py
├── presentation/
│   ├── api/
│   │   └── order_controller.py
│   └── cli/
│       └── order_commands.py
└── tests/
    ├── domain/
    ├── application/
    ├── infrastructure/
    └── integration/
```

## ⚠️ Anti-Patterns to Avoid

1. **Anemic Domain Models**: Domain objects with only getters/setters
2. **Service Layer Bloat**: Putting all logic in service classes
3. **Infrastructure Leak**: Domain depending on specific frameworks
4. **Test Absence**: Generating code without corresponding tests
5. **Mixed Concerns**: Business logic scattered across layers

## 🚀 Advanced Patterns

### Event-Driven Communication
```python
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass(frozen=True)
class DomainEvent(ABC):
    aggregate_id: str
    occurred_at: datetime

@dataclass(frozen=True)
class OrderPlacedEvent(DomainEvent):
    order_total: float
    customer_id: str

class EventStore(ABC):
    @abstractmethod
    def append(self, events: List[DomainEvent]) -> None:
        pass
```

### CQRS Implementation
```python
# Command side
class CreateOrderCommand:
    customer_id: str
    items: List[OrderItem]

class CommandHandler:
    def handle(self, command: CreateOrderCommand) -> None:
        # Process command, update state
        pass

# Query side
class OrderSummaryQuery:
    customer_id: str
    date_range: DateRange

class QueryHandler:
    def handle(self, query: OrderSummaryQuery) -> OrderSummaryDTO:
        # Return read-optimized data
        pass
```

## 📚 Required Reading

When using this skill, Claude should be familiar with:
- Eric Evans' "Domain-Driven Design"
- Robert C. Martin's "Clean Architecture"
- Alistair Cockburn's "Hexagonal Architecture"
- Martin Fowler's "Patterns of Enterprise Application Architecture"

## 🎖️ Certification Criteria

Code generated with this skill must:
1. Pass architectural fitness functions
2. Maintain clear separation between layers
3. Have comprehensive test coverage
4. Include intent documentation
5. Follow all five non-negotiable rules

---

**Note**: This skill is designed to enforce architectural discipline in AI-generated code. All patterns and principles should be applied pragmatically based on project scale and requirements.
