# Architectural Principles and Code Generation Standards — 2026 Edition

## Overview

This skill enforces strict architectural principles and code generation standards for AI-assisted enterprise development. When active, Claude must generate code that adheres to these principles without exception. This edition extends the foundational 2025 patterns with MCP integration, agentic orchestration, parallelized workflows, and AI-native observability — reflecting the reality that most production code is now co-authored with AI.

---

## 🏗️ Six Core Architectural Principles

### 1. Separation of Concerns (SoC)
- **Principle**: Each module/component should have a single, well-defined responsibility
- **Implementation**:
  - Separate data access, business logic, and presentation layers
  - Use dependency injection to manage dependencies
  - Create focused, single-purpose classes and functions
  - Avoid mixing concerns (e.g., UI logic in data access layer)
  - AI-generated code must respect layer boundaries — no "convenience shortcuts" that collapse layers

### 2. Domain-Driven Design (DDD)
- **Principle**: Software design should reflect the business domain
- **Implementation**:
  - Create rich domain models that encapsulate business rules
  - Use ubiquitous language from the business domain
  - Implement aggregates, entities, and value objects
  - Define clear bounded contexts
  - Use domain events for cross-boundary communication
  - MCP servers should align to bounded contexts (one server per context where practical)

### 3. Clean/Hexagonal Architecture
- **Principle**: Business logic should be independent of frameworks, infrastructure, and AI tooling
- **Implementation**:
  ```
  Domain Layer (Core)
  ├── Entities
  ├── Value Objects
  ├── Domain Services
  └── Repository Interfaces (Ports)
  
  Application Layer
  ├── Use Cases / Command Handlers
  ├── DTOs & Structured Schemas
  ├── Application Services
  └── Agent Orchestration Interfaces
  
  Infrastructure Layer
  ├── Database Implementations (Adapters)
  ├── MCP Server Adapters
  ├── External Service Adapters
  ├── AI Provider Adapters (Claude, Gemini, OpenAI)
  └── Framework-specific Code
  
  Presentation Layer
  ├── Controllers / Handlers
  ├── View Models
  ├── UI Components
  └── MCP Client Interfaces
  ```

### 4. High Cohesion, Low Coupling
- **Principle**: Related functionality should be grouped together, dependencies minimized
- **Implementation**:
  - Group related functionality in modules
  - Use interfaces/protocols to define contracts
  - Minimize dependencies between modules
  - Favor composition over inheritance
  - MCP tools should be cohesive — each server exposes a focused capability surface

### 5. MCP-Native Integration Architecture (New for 2026)
- **Principle**: External capabilities should be exposed and consumed through the Model Context Protocol standard
- **Implementation**:
  - Wrap infrastructure services as MCP servers with typed tool schemas
  - Use MCP resources for read-heavy context (database views, config, documents)
  - Use MCP tools for write operations and side-effectful actions
  - Define MCP prompts for reusable interaction patterns
  - All AI ↔ infrastructure boundaries go through MCP — never raw API calls from agent code

### 6. Parallelism-First Design (New for 2026)
- **Principle**: Workflows should be designed for concurrent execution by default, serialized only when dependencies require it
- **Implementation**:
  - Decompose tasks into independent units that can execute concurrently
  - Use DAG-based orchestration for multi-step workflows
  - AI agent subtasks should fan out where no data dependency exists
  - Apply backpressure and rate limiting at the orchestration layer, not within individual tasks

---

## 🛡️ Seven Non-Negotiable Rules

### Rule 1: Zero Business Logic in Infrastructure Components
```python
# ❌ WRONG — Business logic in repository
class UserRepository:
    def get_active_premium_users(self):
        users = self.db.query("SELECT * FROM users")
        return [u for u in users if u.is_active and u.subscription == 'premium']

# ✅ CORRECT — Business logic in domain service
class UserRepository:
    def get_all_users(self) -> list[User]:
        return self.db.query("SELECT * FROM users")

class UserDomainService:
    def get_active_premium_users(self, repository: UserRepositoryPort) -> list[User]:
        users = repository.get_all_users()
        return [u for u in users if self._is_active_premium(u)]
    
    def _is_active_premium(self, user: User) -> bool:
        return user.is_active and user.subscription == 'premium'
```

### Rule 2: Interface-First Development (Ports and Adapters)
```python
from abc import ABC, abstractmethod
from typing import Protocol

# Define ports first — these live in domain/application layer
class PaymentGatewayPort(Protocol):
    def process_payment(self, amount: float, currency: str) -> PaymentResult: ...

class NotificationPort(Protocol):
    def send_notification(self, recipient: str, message: str) -> bool: ...

# Adapters live in infrastructure layer
class StripePaymentAdapter:
    """Implements PaymentGatewayPort via Stripe SDK."""
    def process_payment(self, amount: float, currency: str) -> PaymentResult:
        # Stripe-specific implementation
        ...

# MCP adapter — same port, different transport
class MCPPaymentAdapter:
    """Implements PaymentGatewayPort via MCP tool call to payment-service server."""
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
    
    def process_payment(self, amount: float, currency: str) -> PaymentResult:
        result = self.client.call_tool(
            server="payment-service",
            tool="process_payment",
            arguments={"amount": amount, "currency": currency}
        )
        return PaymentResult.from_mcp_response(result)
```

### Rule 3: Immutable Domain Models
```python
from dataclasses import dataclass, replace, field
from datetime import datetime, UTC

@dataclass(frozen=True)
class Order:
    id: str
    customer_id: str
    total_amount: float
    status: str
    created_at: datetime
    domain_events: tuple[DomainEvent, ...] = field(default=())
    
    def mark_as_paid(self) -> 'Order':
        return replace(
            self,
            status='PAID',
            domain_events=self.domain_events + (
                OrderPaidEvent(aggregate_id=self.id, occurred_at=datetime.now(UTC)),
            )
        )
    
    def apply_discount(self, percentage: float) -> 'Order':
        if percentage < 0 or percentage > 100:
            raise DomainError("Discount must be between 0 and 100")
        new_amount = self.total_amount * (1 - percentage / 100)
        return replace(self, total_amount=round(new_amount, 2))
```

### Rule 4: Mandatory Testing Coverage
```python
import pytest
from unittest.mock import AsyncMock

# Domain model test — no mocks needed, pure logic
class TestOrder:
    def test_mark_as_paid_creates_new_instance(self):
        order = Order(id="1", customer_id="C1", total_amount=100.0,
                     status="PENDING", created_at=datetime.now(UTC))
        paid_order = order.mark_as_paid()
        
        assert paid_order.status == "PAID"
        assert order.status == "PENDING"  # Original unchanged
        assert paid_order is not order
        assert len(paid_order.domain_events) == 1

# Use case test — mock ports, verify orchestration
class TestCreateOrderUseCase:
    @pytest.mark.asyncio
    async def test_create_order_with_valid_data(self):
        repository = AsyncMock(spec=OrderRepositoryPort)
        event_bus = AsyncMock(spec=EventBusPort)
        use_case = CreateOrderUseCase(repository=repository, event_bus=event_bus)
        
        result = await use_case.execute(customer_id="C1", items=[...])
        
        repository.save.assert_awaited_once()
        event_bus.publish.assert_awaited()
        assert result.is_success

# MCP integration test — verify tool schema compliance
class TestPaymentMCPServer:
    async def test_process_payment_tool_schema(self):
        server = PaymentMCPServer()
        tools = await server.list_tools()
        
        payment_tool = next(t for t in tools if t.name == "process_payment")
        assert "amount" in payment_tool.input_schema["properties"]
        assert "currency" in payment_tool.input_schema["properties"]
```

### Rule 5: Documentation of Architectural Intent
```python
"""
Order Processing Module

Architectural Intent:
- Order lifecycle management following DDD principles
- Order aggregate is the consistency boundary
- All state changes go through domain methods to enforce invariants
- External payment processing abstracted behind PaymentGatewayPort
- Domain events published for cross-context communication

MCP Integration:
- Exposed as 'order-service' MCP server with tools: create_order, 
  update_status, cancel_order
- Resources: order://list, order://{id} for read access
- Consumes: payment-service, notification-service MCP servers

Parallelization Notes:
- Order validation and inventory check run concurrently
- Payment and notification are sequential (payment must succeed first)

Key Design Decisions:
1. Orders are immutable — state changes produce new instances
2. Payment processing is behind a port (swappable Stripe/MCP adapter)
3. Status transitions validated via state machine in domain model
4. Pricing logic delegated to PricingDomainService
"""
```

### Rule 6: MCP-Compliant Service Boundaries (New for 2026)
```python
from mcp.server import Server
from mcp.types import Tool, Resource

class OrderMCPServer:
    """
    MCP server exposing order domain capabilities.
    
    Each bounded context should have exactly one MCP server.
    Tools = write operations, Resources = read operations.
    """
    
    def __init__(self, create_order_use_case: CreateOrderUseCase,
                 query_service: OrderQueryService):
        self.server = Server("order-service")
        self._create_order = create_order_use_case
        self._query = query_service
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        @self.server.tool()
        async def create_order(customer_id: str, items: list[dict]) -> dict:
            """Create a new order. Validates inventory and calculates pricing."""
            result = await self._create_order.execute(customer_id, items)
            return result.to_dict()
        
        @self.server.tool()
        async def cancel_order(order_id: str, reason: str) -> dict:
            """Cancel an existing order. Only pending orders can be cancelled."""
            ...
    
    def _register_resources(self):
        @self.server.resource("order://{order_id}")
        async def get_order(order_id: str) -> str:
            """Read-only access to order data."""
            order = await self._query.get_by_id(order_id)
            return order.to_json()
```

### Rule 7: Parallel-Safe Orchestration (New for 2026)
```python
import asyncio
from dataclasses import dataclass

@dataclass
class WorkflowStep:
    name: str
    execute: Callable
    depends_on: list[str] = field(default_factory=list)

class DAGOrchestrator:
    """
    Executes workflow steps respecting dependency order,
    parallelizing independent steps automatically.
    """
    
    def __init__(self, steps: list[WorkflowStep]):
        self.steps = {s.name: s for s in steps}
        self._validate_no_cycles()
    
    async def execute(self, context: dict) -> dict:
        completed: dict[str, Any] = {}
        pending = set(self.steps.keys())
        
        while pending:
            # Find all steps whose dependencies are satisfied
            ready = [
                name for name in pending
                if all(dep in completed for dep in self.steps[name].depends_on)
            ]
            if not ready:
                raise OrchestrationError("Circular dependency detected")
            
            # Execute ready steps concurrently
            results = await asyncio.gather(
                *(self.steps[name].execute(context, completed) for name in ready),
                return_exceptions=True
            )
            
            for name, result in zip(ready, results):
                if isinstance(result, Exception):
                    raise OrchestrationError(f"Step {name} failed: {result}")
                completed[name] = result
                pending.discard(name)
        
        return completed

# Usage in a use case
class ProcessOrderUseCase:
    async def execute(self, order: Order) -> ProcessingResult:
        orchestrator = DAGOrchestrator([
            WorkflowStep("validate", self._validate_order),
            WorkflowStep("check_inventory", self._check_inventory),
            WorkflowStep("calculate_pricing", self._calculate_pricing),
            # These two run in parallel — no dependency between them
            WorkflowStep("reserve_stock", self._reserve_stock, 
                        depends_on=["validate", "check_inventory"]),
            WorkflowStep("process_payment", self._process_payment, 
                        depends_on=["validate", "calculate_pricing"]),
            # This waits for both parallel branches
            WorkflowStep("confirm_order", self._confirm_order, 
                        depends_on=["reserve_stock", "process_payment"]),
        ])
        return await orchestrator.execute({"order": order})
```

---

## 📋 Implementation Checklist

When generating code, Claude must verify:

### Layer Separation
- [ ] Domain layer has ZERO dependencies on infrastructure or AI providers
- [ ] Application layer depends only on domain layer
- [ ] Infrastructure layer implements interfaces defined in domain
- [ ] Presentation layer only interacts with application layer
- [ ] MCP servers live in infrastructure, consuming application-layer use cases

### Interface Design
- [ ] All external dependencies have interface definitions (Protocols or ABCs)
- [ ] Interfaces are defined in the domain/application layer
- [ ] Concrete implementations are in infrastructure layer
- [ ] Dependency injection wires implementations at composition root
- [ ] MCP tool schemas mirror port method signatures

### Domain Modeling
- [ ] Domain models are immutable (frozen dataclasses or equivalent)
- [ ] Business rules are encapsulated in domain objects
- [ ] Value objects are used for concepts without identity
- [ ] Aggregates maintain consistency boundaries
- [ ] Domain events are collected and dispatched, not fired inline

### MCP Compliance
- [ ] Each bounded context has at most one MCP server
- [ ] Tools are used for write operations (commands)
- [ ] Resources are used for read operations (queries)
- [ ] MCP tool schemas include clear descriptions and type annotations
- [ ] Error responses use structured MCP error format

### Parallelization
- [ ] Independent operations are identified and grouped for concurrent execution
- [ ] Dependencies between steps are explicit (DAG or similar)
- [ ] Shared mutable state is eliminated or synchronized
- [ ] Backpressure/rate limiting is applied at orchestrator level
- [ ] Timeout and cancellation propagation is implemented

### Testing Requirements
- [ ] Unit tests for all domain logic (no mocks — pure functions)
- [ ] Use case tests with mocked ports
- [ ] MCP schema compliance tests
- [ ] Integration tests for infrastructure adapters
- [ ] Parallel workflow tests verifying correct execution order
- [ ] Test coverage meets minimum 80% threshold

### Documentation Standards
- [ ] Each module has architectural intent documented
- [ ] MCP integration points are explicitly described
- [ ] Parallelization strategy is documented per workflow
- [ ] Key design decisions are recorded
- [ ] Domain concepts use ubiquitous language

---

## 🎯 Code Generation Guidelines

### When Creating New Components

1. **Start with the Domain**
   - Define entities and value objects first
   - Identify aggregate boundaries
   - Document invariants and business rules
   - Define domain events for state transitions

2. **Define Interfaces (Ports)**
   - Create port interfaces for all external dependencies
   - Design application service interfaces
   - Keep interfaces focused and cohesive
   - Use Python `Protocol` for structural typing

3. **Implement Use Cases**
   - One use case per class
   - Orchestrate domain objects via ports
   - Use DAG orchestration for multi-step workflows
   - Parallelize independent steps by default

4. **Add Infrastructure Adapters**
   - Implement adapters for each port
   - Build MCP servers wrapping use cases
   - Build MCP client adapters for consumed services
   - Configure dependency injection at composition root

5. **Create Tests (at every layer)**
   - Domain: pure unit tests, no mocks
   - Application: use case tests with mocked ports
   - Infrastructure: integration tests with real dependencies
   - MCP: schema compliance + round-trip tests
   - Orchestration: verify parallel execution and failure modes

### Example Project Structure (2026)
```
project/
├── domain/
│   ├── entities/
│   │   ├── order.py
│   │   └── customer.py
│   ├── value_objects/
│   │   ├── money.py
│   │   └── address.py
│   ├── events/
│   │   ├── order_events.py
│   │   └── event_base.py
│   ├── services/
│   │   └── pricing_service.py
│   └── ports/
│       ├── repository_ports.py
│       ├── payment_ports.py
│       └── notification_ports.py
├── application/
│   ├── commands/
│   │   ├── create_order.py
│   │   └── cancel_order.py
│   ├── queries/
│   │   ├── get_order.py
│   │   └── list_orders.py
│   ├── orchestration/
│   │   └── process_order_workflow.py
│   └── dtos/
│       └── order_dto.py
├── infrastructure/
│   ├── repositories/
│   │   └── order_repository.py
│   ├── adapters/
│   │   ├── stripe_payment_adapter.py
│   │   ├── email_notification_adapter.py
│   │   └── mcp_payment_adapter.py
│   ├── mcp_servers/
│   │   ├── order_server.py
│   │   └── server_config.json
│   ├── mcp_clients/
│   │   └── payment_client.py
│   └── config/
│       ├── dependency_injection.py
│       └── mcp_registry.py
├── presentation/
│   ├── api/
│   │   └── order_controller.py
│   └── cli/
│       └── order_commands.py
└── tests/
    ├── domain/
    │   └── test_order.py
    ├── application/
    │   ├── test_create_order.py
    │   └── test_process_order_workflow.py
    ├── infrastructure/
    │   ├── test_order_repository.py
    │   └── test_mcp_servers.py
    └── integration/
        └── test_order_flow.py
```

---

## 🔌 MCP Integration Patterns

### Pattern 1: Bounded Context as MCP Server
Each bounded context exposes its capabilities as an MCP server. This is the primary integration pattern for microservices and modular monoliths.

```python
# infrastructure/mcp_servers/order_server.py
from mcp.server import Server
from mcp.types import Tool, Resource, Prompt

def create_order_server(
    container: DependencyContainer
) -> Server:
    server = Server("order-service")
    
    # TOOLS — write operations (commands)
    @server.tool()
    async def create_order(customer_id: str, items: list[dict]) -> dict:
        """Create a new order with validation and pricing."""
        use_case = container.resolve(CreateOrderUseCase)
        return (await use_case.execute(customer_id, items)).to_dict()
    
    # RESOURCES — read operations (queries)  
    @server.resource("order://orders/{order_id}")
    async def get_order(order_id: str) -> str:
        query = container.resolve(GetOrderQuery)
        return (await query.execute(order_id)).to_json()
    
    @server.resource("order://orders")
    async def list_orders() -> str:
        query = container.resolve(ListOrdersQuery)
        return (await query.execute()).to_json()
    
    # PROMPTS — reusable interaction patterns
    @server.prompt()
    async def order_summary(order_id: str) -> str:
        """Generate a human-readable order summary for agent consumption."""
        order = await container.resolve(GetOrderQuery).execute(order_id)
        return f"""Order {order.id}:
        Customer: {order.customer_id}
        Status: {order.status}
        Total: {order.total_amount}
        Items: {len(order.items)}"""
    
    return server
```

### Pattern 2: MCP Client as Infrastructure Adapter
When your service consumes another bounded context, wrap the MCP client call behind a port.

```python
# infrastructure/mcp_clients/payment_client.py
from domain.ports.payment_ports import PaymentGatewayPort

class MCPPaymentClient(PaymentGatewayPort):
    """Adapter that calls the payment-service MCP server."""
    
    def __init__(self, mcp_session: ClientSession):
        self.session = mcp_session
    
    async def process_payment(self, amount: float, currency: str) -> PaymentResult:
        result = await self.session.call_tool(
            "process_payment",
            arguments={"amount": amount, "currency": currency}
        )
        return PaymentResult.from_dict(result.content)
```

### Pattern 3: MCP Server Registry
Centralized configuration for which MCP servers are available and how they connect.

```json
{
  "mcpServers": {
    "order-service": {
      "command": "python",
      "args": ["-m", "infrastructure.mcp_servers.order_server"],
      "env": { "DATABASE_URL": "${DATABASE_URL}" }
    },
    "payment-service": {
      "command": "python",
      "args": ["-m", "infrastructure.mcp_servers.payment_server"],
      "env": { "STRIPE_KEY": "${STRIPE_SECRET_KEY}" }
    },
    "notification-service": {
      "url": "https://notifications.internal/mcp",
      "transport": "streamable-http"
    }
  }
}
```

---

## ⚡ Parallelization Patterns

### Pattern 1: Fan-Out / Fan-In for Independent Operations
```python
async def enrich_customer_profile(customer_id: str) -> EnrichedProfile:
    """Fetch data from multiple sources concurrently, merge results."""
    credit_check, order_history, preferences = await asyncio.gather(
        credit_service.check(customer_id),
        order_query.get_history(customer_id),
        preference_service.get(customer_id),
    )
    return EnrichedProfile(
        credit_score=credit_check.score,
        total_orders=len(order_history),
        preferences=preferences,
    )
```

### Pattern 2: Pipeline Parallelism with Typed Stages
```python
from typing import TypeVar, Generic

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")

class PipelineStage(Generic[T_In, T_Out], ABC):
    @abstractmethod
    async def process(self, input: T_In) -> T_Out: ...

class ParallelPipeline:
    """Processes items through stages, parallelizing within each stage."""
    
    def __init__(self, stages: list[PipelineStage], max_concurrency: int = 10):
        self.stages = stages
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def execute(self, items: list) -> list:
        results = items
        for stage in self.stages:
            results = await asyncio.gather(
                *(self._run_with_limit(stage.process, item) for item in results)
            )
        return results
    
    async def _run_with_limit(self, fn, item):
        async with self.semaphore:
            return await fn(item)
```

### Pattern 3: Agent Task Decomposition
```python
class AgentOrchestrator:
    """
    Decomposes complex tasks into parallel subtasks for AI agents.
    Each subtask gets its own context and MCP server access.
    """
    
    async def execute_plan(self, plan: ExecutionPlan) -> dict:
        # Group tasks by dependency level
        levels = plan.topological_sort()
        
        results = {}
        for level in levels:
            # All tasks at the same level run concurrently
            level_results = await asyncio.gather(
                *(self._execute_task(task, results) for task in level),
                return_exceptions=True,
            )
            for task, result in zip(level, level_results):
                if isinstance(result, Exception):
                    if task.is_critical:
                        raise OrchestrationError(f"Critical task failed: {task.name}")
                    results[task.name] = TaskResult.failed(str(result))
                else:
                    results[task.name] = result
        
        return results
```

---

## 🧠 AI-Native Context Patterns

### Pattern 1: Structured Output Schemas
Always define explicit schemas for AI-generated structured data.

```python
from pydantic import BaseModel, Field

class OrderAnalysis(BaseModel):
    """Schema for AI-generated order analysis — used as structured output."""
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Fraud risk assessment"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of risk assessment")
    recommended_actions: list[str] = Field(default_factory=list)

# Use as structured output in AI calls
analysis = await ai_client.generate(
    prompt=f"Analyze this order for fraud risk: {order.to_json()}",
    response_schema=OrderAnalysis,
)
```

### Pattern 2: Context Window Management
When building AI-powered features, manage context explicitly.

```python
class ContextBuilder:
    """
    Builds optimized context for AI operations.
    Respects token limits and prioritizes relevant information.
    """
    
    def __init__(self, max_tokens: int = 100_000):
        self.max_tokens = max_tokens
        self.sections: list[ContextSection] = []
    
    def add_system(self, content: str, priority: int = 0) -> 'ContextBuilder':
        self.sections.append(ContextSection("system", content, priority))
        return self
    
    def add_domain_context(self, entities: list, priority: int = 1) -> 'ContextBuilder':
        serialized = "\n".join(e.to_context_string() for e in entities)
        self.sections.append(ContextSection("domain", serialized, priority))
        return self
    
    def build(self) -> list[dict]:
        # Sort by priority, trim to fit token budget
        sorted_sections = sorted(self.sections, key=lambda s: s.priority)
        messages = []
        remaining_tokens = self.max_tokens
        for section in sorted_sections:
            tokens = estimate_tokens(section.content)
            if tokens <= remaining_tokens:
                messages.append(section.to_message())
                remaining_tokens -= tokens
        return messages
```

### Pattern 3: Persistent Memory via MCP
Use MCP resources to give AI agents access to persistent project context.

```python
# MCP server that exposes project memory as resources
@server.resource("memory://decisions")
async def get_decisions() -> str:
    """Architectural decisions made in this project."""
    return load_adr_documents()

@server.resource("memory://conventions")
async def get_conventions() -> str:
    """Code conventions and patterns used in this project."""
    return load_conventions_doc()

@server.resource("memory://glossary")
async def get_glossary() -> str:
    """Domain glossary — ubiquitous language definitions."""
    return load_glossary()
```

---

## ⚠️ Anti-Patterns to Avoid

1. **Anemic Domain Models**: Domain objects with only getters/setters and no behavior
2. **Service Layer Bloat**: All logic dumped into service classes, domain models are empty shells
3. **Infrastructure Leak**: Domain layer importing `sqlalchemy`, `stripe`, or any framework
4. **Test Absence**: Generating code without corresponding tests
5. **Mixed Concerns**: Business logic scattered across layers
6. **MCP Sprawl**: One MCP server per function instead of per bounded context
7. **Synchronous Bottleneck**: Sequential execution of independent operations that could be parallelized
8. **God Orchestrator**: A single orchestration layer that knows about every domain detail instead of delegating
9. **Untyped AI Output**: Using raw string responses from AI without schema validation
10. **Context Stuffing**: Dumping entire databases into AI context instead of curating relevant information

---

## 🚀 Advanced Patterns

### Event-Driven Communication
```python
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Protocol

@dataclass(frozen=True)
class DomainEvent:
    aggregate_id: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

@dataclass(frozen=True)
class OrderPlacedEvent(DomainEvent):
    order_total: float = 0.0
    customer_id: str = ""

class EventBusPort(Protocol):
    async def publish(self, events: list[DomainEvent]) -> None: ...
    async def subscribe(self, event_type: type, handler: Callable) -> None: ...

# MCP-backed event bus adapter
class MCPEventBusAdapter:
    """Publishes domain events to an event-bus MCP server."""
    
    async def publish(self, events: list[DomainEvent]) -> None:
        for event in events:
            await self.mcp_session.call_tool(
                "publish_event",
                arguments={
                    "event_type": type(event).__name__,
                    "payload": asdict(event),
                }
            )
```

### CQRS with Separate Read/Write MCP Interfaces
```python
# Command side — MCP tools
@server.tool()
async def create_order(customer_id: str, items: list[dict]) -> dict:
    """Write path: validates, persists, publishes events."""
    handler = container.resolve(CreateOrderCommandHandler)
    return await handler.handle(CreateOrderCommand(customer_id=customer_id, items=items))

# Query side — MCP resources (read-optimized, potentially from a different store)
@server.resource("order://summaries/{customer_id}")
async def order_summaries(customer_id: str) -> str:
    """Read path: returns pre-computed summary from read model."""
    handler = container.resolve(OrderSummaryQueryHandler)
    return (await handler.handle(OrderSummaryQuery(customer_id=customer_id))).to_json()
```

### Multi-Agent Coordination
```python
class MultiAgentCoordinator:
    """
    Coordinates multiple AI agents working on related subtasks.
    Each agent gets scoped MCP access and shared context.
    """
    
    async def coordinate(self, task: ComplexTask) -> CoordinatedResult:
        subtasks = self.decompose(task)
        
        # Phase 1: Independent research (parallel)
        research_results = await asyncio.gather(
            *(self.spawn_agent(st, role="researcher") for st in subtasks)
        )
        
        # Phase 2: Synthesis (single agent with all context)
        synthesis = await self.spawn_agent(
            SynthesisTask(inputs=research_results),
            role="synthesizer",
        )
        
        # Phase 3: Validation (parallel — each validator checks one aspect)
        validations = await asyncio.gather(
            self.spawn_agent(synthesis, role="accuracy_checker"),
            self.spawn_agent(synthesis, role="consistency_checker"),
            self.spawn_agent(synthesis, role="completeness_checker"),
        )
        
        return CoordinatedResult(
            output=synthesis,
            validations=validations,
        )
```

---

## 📚 Required Reading

When using this skill, Claude should be familiar with:
- Eric Evans' "Domain-Driven Design"
- Robert C. Martin's "Clean Architecture"
- Alistair Cockburn's "Hexagonal Architecture"
- Martin Fowler's "Patterns of Enterprise Application Architecture"
- **Model Context Protocol Specification** (modelcontextprotocol.io)
- **Anthropic's Agent Design Patterns** (docs.anthropic.com)

---

## 🎖️ Certification Criteria

Code generated with this skill must:
1. Pass architectural fitness functions (layer dependency checks)
2. Maintain clear separation between all layers
3. Have comprehensive test coverage (≥80%, domain layer ≥95%)
4. Include architectural intent documentation
5. Follow all seven non-negotiable rules
6. Expose/consume external capabilities via MCP where applicable
7. Parallelize independent operations by default
8. Use typed schemas for all AI-generated structured output

---

**Note**: This skill enforces architectural discipline in AI-assisted code generation. All patterns should be applied pragmatically based on project scale — a solo microservice doesn't need multi-agent coordination, but it still needs clean layer separation and tests. Scale the ceremony to the complexity.
